"""This module contains the implementation of the analysis processor. The processor is responsible for executing the analysis tasks in parallel and caching the results."""

import logging
import sys
import threading
from collections import OrderedDict, namedtuple
if sys.version_info >= (3, 3):
    from collections.abc import Iterable
else:
    from collections import Iterable
from functools import partial
from typing import List, Union, Mapping, Tuple, Any
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from threading import RLock, Condition
from queue import Queue, Empty

from cachetools import Cache
from bidict import bidict

from vot import ToolkitException
from vot.dataset import Sequence
from vot.tracker import Tracker
from vot.experiment import Experiment
from vot.analysis import SeparableAnalysis, Analysis
from vot.utilities import arg_hash, class_fullname, Progress
from vot.utilities.data import Grid

logger = logging.getLogger("vot")

def hashkey(analysis: Analysis, *args):
    """Compute a hash key for the analysis and its arguments. The key is used for caching the results."""
    def transform(arg):
        """Transform an argument into a hashable object."""
        if isinstance(arg, Sequence):
            return arg.name
        if isinstance(arg, Tracker):
            return arg.reference
        if isinstance(arg, Experiment):
            return arg.identifier
        if isinstance(arg, Mapping):
            return arg_hash(**{k: transform(v) for k, v in arg.items()})
        if isinstance(arg, Iterable):
            return arg_hash(*[transform(i) for i in arg])

    return (analysis.identifier, *[transform(arg) for arg in args])

def unwrap(arg):
    """Unwrap a single element list."""

    if isinstance(arg, list) and len(arg) == 1:
        return arg[0]
    else:
        return arg

class AnalysisError(ToolkitException):
    """An exception that is raised when an analysis fails."""

    def __init__(self, cause, task=None):
        """Creates an analysis error.
        
        Args:
            cause (Exception): The cause of the error.
            task (AnalysisTask, optional): The task that caused the error. Defaults to None.
            
        """
        self._tasks = []
        self._cause = cause
        super().__init__(cause, task)
        self._tasks.append(task)

    @property
    def task(self):
        """The task that caused the error."""
        return self._tasks[-1]

    def __str__(self):
        """String representation of the error."""
        return "Error during analysis {}".format(self.task)

    def print(self, logoutput):
        """Print the error to the log output."""
        logoutput.error(str(self))
        if len(self._tasks) > 1:
            for task in reversed(self._tasks[:-1]):
                logoutput.debug("Caused by an error in subtask: %s", str(task))
        logoutput.exception(self.__cause__)

    @property
    def root_cause(self):
        """The root cause of the error."""
        cause = self._cause
        if cause is None:
            return None
        if isinstance(cause, AnalysisError):
            return cause.root_cause
        else:
            return cause

class DebugExecutor(Executor):
    """A synchronous executor used for debugging. Do not use it in practice.
    """

    Task = namedtuple("Task", ["fn", "args", "kwargs", "promise"])

    def __init__(self, strict=True):
        """Creates a single-thread debug executor.

        Args:
            strict (bool, optional): Strict mode means that the executor stops if any of the tasks fails. Defaults to True.
        """
        self._queue = Queue()
        self._lock = threading.RLock()
        self._semaphor = threading.Condition(self._lock)
        self._thread = threading.Thread(target=self._run)
        self._alive = True
        self._strict = True
        self._thread.start()

    def submit(self, fn, *args, **kwargs):
        """Submits a task to the executor."""

        promise = Future()
        with self._lock:
            self._queue.put(DebugExecutor.Task(fn, args, kwargs, promise))
            self._semaphor.notify()
            logger.debug("Adding task %s to queue", fn)
            return promise

    def _run(self):
        """The main loop of the executor."""

        while True:

            with self._lock:
                if not self._alive:
                    break

                try:
                    task = self._queue.get(False)
                except Empty:
                    self._semaphor.wait()
                    continue

            if task.promise.cancelled():
                logger.debug("Task %s cancelled, skipping", task.fn)
                continue

            error = None

            try:

                logger.debug("Running task %s", task.fn)
                result = task.fn(*task.args, **task.kwargs)
                task.promise.set_result(result)
                logger.debug("Task %s completed", task.fn)


            except TypeError as e:
                logger.info("Task %s call resulted in error: %s", task.fn, e)

                error = e

            except Exception as e:

                error = e

                logger.info("Task %s resulted in exception: %s", task.fn, e)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.exception(e)

                logger.exception(e)

            if error is not None:
                task.promise.set_exception(error)

                if self._strict:
                    self._alive = False
                    self._clear()
                    break

    def _clear(self):
        """Clears the queue."""
        with self._lock:

            while True:
                try:
                    task = self._queue.get(False)
                    if not task.promise.done():
                        task.promise.cancel()
                except Empty:
                    break

    def shutdown(self, wait=True):
        """Shuts down the executor. If wait is True, the method blocks until all tasks are completed. 
        
        Args:
            wait (bool, optional): Wait for all tasks to complete. Defaults to True.
        """

        self._alive = False
        self._clear()
        if wait:
            with self._lock:
                self._semaphor.notify()

            self._thread.join()

class ExecutorWrapper(object):
    """A wrapper for an executor that allows to submit tasks with dependencies."""

    def __init__(self, executor: Executor):
        """Creates an executor wrapper.
        
        Args:
            executor (Executor): The executor to wrap.
        """
        self._lock = RLock()
        self._executor = executor
        self._pending = OrderedDict()
        self._total = 0

    @property
    def total(self):
        """The total number of tasks submitted to the executor."""
        return self._total

    def submit(self, fn, *futures: Tuple[Future], mapping=None) -> Future:
        """Submits a task to the executor. The task will be executed when all futures are completed. 

        Args:
            fn (Callable): The task to execute.
            futures (Tuple[Future]): The futures that must be completed before the task is executed.
            mapping (Dict[Future, Any], optional): A mapping of futures to values. Defaults to None.

        Returns:
            Future: A future that will be completed when the task is completed.
        """

        with self._lock:

            self._total += 1

            if len(futures) == 0:
                return self._executor.submit(fn)
            else:
                depend = FuturesAggregator(*futures)

            proxy = Future()
            self._pending[proxy] = depend

            proxy.add_done_callback(self._proxy_done)
            depend.add_done_callback(partial(self._ready_callback, fn, mapping, proxy))

            return proxy

    def _ready_callback(self, fn, mapping, proxy: Future, future: Future):
        """ Internally handles completion of dependencies. Submits the task to the executor.
        """

        with self._lock:

            if not proxy in self._pending:
                return

            del self._pending[proxy]

            if future.cancelled():
                proxy.cancel()
            if not proxy.set_running_or_notify_cancel():
                return
            exception = future.exception()
            if exception is not None:
                proxy.set_exception(exception)
                return
        
            if mapping is None:
                dependencies = future.result()
            else:
                dependencies = mapping(*future.result())

            internal = self._executor.submit(fn, *dependencies)
            internal.add_done_callback(partial(self._done_callback, proxy))

    def _done_callback(self, proxy: Future, future: Future):
        """ Internally handles completion of executor future, copies result to proxy

        Args:
            proxy (Future): Proxy future
            future (Future): Executor future
        """

        if future.cancelled():
            proxy.cancel()
        exception = future.exception()
        if exception is not None:
            proxy.set_exception(exception)
        else:
            result = future.result()
            proxy.set_result(result)


    def _proxy_done(self, future: Future):
        """ Internally handles events for proxy futures, this means handling cancellation.

        Args:
            future (Future): Proxy future

        """

        with self._lock:

            if not future in self._pending:
                return

            dependency = self._pending[future]

            del self._pending[future]

            if future.cancelled():
                dependency.cancel()

class FuturesAggregator(Future):
    """A future that aggregates results from other futures."""

    def __init__(self, *futures: Tuple[Future]):
        """Initializes the aggregator.

        Args:
            *futures (Tuple[Future]): The futures to aggregate.
        """

        super().__init__()
        self._lock = RLock()
        self._results = [None] * len(futures)
        self._tasks = list(futures)

        for i, future in enumerate(futures):
            future.add_done_callback(partial(self._on_result, i))

        if not self._results:
            self.set_result([])

    def _on_result(self, i, future):
        """Handles completion of a dependency future."""

        with self._lock:
            if self.done():
                return
            try:
                self._results[i] = future.result()
            except Exception as e:
                self.set_exception(e)
                return

            if all([x is not None for x in self._results]):
                self.set_result(self._results)

    def _on_done(self, future):
        """Handles completion of the future."""

        with self._lock:
            try:
                self.set_result(future.result())
            except AnalysisError as e:
                self.set_exception(e)

    def cancel(self):
        """Cancels the future and all dependencies."""

        with self._lock:
            for promise in self._tasks:
                promise.cancel()
            super().cancel()


class AnalysisTask(object):
    """A task that computes an analysis."""

    def __init__(self, analysis: Analysis, experiment: Experiment, 
            trackers: List[Tracker], sequences: List[Sequence]):
        """Initializes a new instance of the AnalysisTask class.
        
        Args:
            analysis (Analysis): The analysis to compute.
            experiment (Experiment): The experiment to compute the analysis for.
            trackers (List[Tracker]): The trackers to compute the analysis for.
            sequences (List[Sequence]): The sequences to compute the analysis for.
        """

        self._analysis = analysis
        self._trackers = trackers
        self._experiment = experiment
        self._sequences = sequences
        self._key = hashkey(analysis, experiment, trackers, sequences)

    def __call__(self, dependencies: List[Grid] = None):
        """Computes the analysis. 

        Args:
            dependencies (List[Grid], optional): The dependencies to use. Defaults to None.

        Returns:
            Grid: The computed analysis.
        """

        try:
            if dependencies is None:
                dependencies = []
            return self._analysis.compute(self._experiment, self._trackers, self._sequences, dependencies)
        except BaseException as e:
            raise AnalysisError(cause=e, task=self._key)

class AnalysisPartTask(object):
    """A task that computes a part of a separable analysis."""

    def __init__(self, analysis: SeparableAnalysis, experiment: Experiment,
            trackers: List[Tracker], sequences: List[Sequence]):
        """Initializes a new instance of the AnalysisPartTask class.
        
        Args:
            analysis (SeparableAnalysis): The analysis to compute.
            experiment (Experiment): The experiment to compute the analysis for.
            trackers (List[Tracker]): The trackers to compute the analysis for.
            sequences (List[Sequence]): The sequences to compute the analysis for.
        """
        self._analysis = analysis
        self._trackers = trackers
        self._experiment = experiment
        self._sequences = sequences
        self._key = hashkey(analysis, experiment, unwrap(trackers), unwrap(sequences))

    def __call__(self, dependencies: List[Grid] = None):
        """Computes the analysis.
        
        Args:
            dependencies (List[Grid], optional): The dependencies to use. Defaults to None.

        Returns:
            Grid: The computed analysis.
        """
        try:
            if dependencies is None:
                dependencies = []
            return self._analysis.compute(self._experiment, self._trackers, self._sequences, dependencies)
        except BaseException as e:
            raise AnalysisError(cause=e, task=self._key)

class AnalysisJoinTask(object):
    """A task that joins the results of a separable analysis."""

    def __init__(self, analysis: SeparableAnalysis, experiment: Experiment,
            trackers: List[Tracker], sequences: List[Sequence]):
        
        """Initializes a new instance of the AnalysisJoinTask class.
        
        Args:
            analysis (Analysis): The analysis to join.
            experiment (Experiment): The experiment to join the analysis for.
            trackers (List[Tracker]): The trackers to join the analysis for.
            sequences (List[Sequence]): The sequences to join the analysis for.    
        """
        self._analysis = analysis
        self._trackers = trackers
        self._experiment = experiment
        self._sequences = sequences
        self._key = hashkey(analysis, experiment, trackers, sequences)

    def __call__(self, results: List[Grid]):
        """Joins the results of the analysis.

        Args:
            results (List[Grid]): The results to join.

        Returns:
            Grid: The joined analysis.
        """

        try:
            return self._analysis.join(self._trackers, self._sequences, results)
        except BaseException as e:
            raise AnalysisError(cause=e, task=self._key)

class AnalysisFuture(Future):
    """A future that represents the result of an analysis."""

    def __init__(self, key):
        """Initializes a new instance of the AnalysisFuture class.

        Args:
            key (str): The key of the analysis.
        """

        super().__init__()
        self._key = key

    @property
    def key(self):
        """Gets the key of the analysis."""
        return self._key

    def __repr__(self) -> str:
        """Gets a string representation of the future."""
        return "<AnalysisFuture key={}>".format(self._key)

class AnalysisProcessor(object):
    """A processor that computes analyses."""

    _context = threading.local()

    def __init__(self, executor: Executor = None, cache: Cache = None):
        """Initializes a new instance of the AnalysisProcessor class.

        Args:
            executor (Executor, optional): The executor to use for computations. Defaults to None.
            cache (Cache, optional): The cache to use for computations. Defaults to None.

        """
        if executor is None:
            executor = ThreadPoolExecutor(1)

        self._executor = ExecutorWrapper(executor)
        self._cache = cache
        self._pending = bidict()
        self._promises = dict()
        self._lock = RLock()
        self._wait_condition = Condition()

    def commit(self, analysis: Analysis, experiment: Experiment,
        trackers: Union[Tracker, List[Tracker]], sequences: Union[Sequence, List[Sequence]]) -> Future:
        """Commits an analysis for computation. If the analysis is already being computed, the existing future is returned.

        Args:
            analysis (Analysis): The analysis to commit.
            experiment (Experiment): The experiment to commit the analysis for.
            trackers (Union[Tracker, List[Tracker]]): The trackers to commit the analysis for.
            sequences (Union[Sequence, List[Sequence]]): The sequences to commit the analysis for.

        Returns:
            Future: A future that represents the result of the analysis.
        """

        key = hashkey(analysis, experiment, trackers, sequences)

        with self._lock:

            promise = self._exists(key)

            if not promise is None:
                return promise
    
            promise = AnalysisFuture(key)
            promise.add_done_callback(self._promise_cancelled)

            dependencies = [self.commit(dependency, experiment, trackers, sequences) for dependency in analysis.dependencies()]

            if isinstance(analysis, SeparableAnalysis):

                def select_dependencies(analysis: SeparableAnalysis, tracker: int, sequence: int, *dependencies):
                    """Selects the dependencies for a part of a separable analysis."""
                    return [analysis.select(meta, data, tracker, sequence) for meta, data in zip(analysis.dependencies(), dependencies)]

                promise = AnalysisFuture(key)
                promise.add_done_callback(self._promise_cancelled)

                parts = analysis.separate(trackers, sequences)
                partpromises = []

                for part in parts:
                    partkey = hashkey(analysis, experiment, unwrap(part.trackers), unwrap(part.sequences))
                
                    partpromise = self._exists(partkey)
                    if not partpromise is None:
                        partpromises.append(partpromise)
                        continue
                
                    partpromise = AnalysisFuture(partkey)
                    partpromises.append(partpromise)

                    executorpromise = self._executor.submit(AnalysisPartTask(analysis, experiment, part.trackers, part.sequences), *dependencies, 
                        mapping=partial(select_dependencies, analysis, part.tid, part.sid))
                    self._promises[partkey] = [partpromise]
                    self._pending[partkey] = executorpromise
                    executorpromise.add_done_callback(self._future_done)

                executorpromise = self._executor.submit(AnalysisJoinTask(analysis, experiment, trackers, sequences),
                        *partpromises, mapping=lambda *x: [list(x)])
                self._pending[key] = executorpromise
            else:
                task = AnalysisTask(analysis, experiment, trackers, sequences)
                executorpromise = self._executor.submit(task, *dependencies, mapping=lambda *x: [list(x)])
                self._pending[key] = executorpromise

            self._promises[key] = [promise]
            executorpromise.add_done_callback(self._future_done)
            logger.debug("Adding analysis task %s", key)

            return promise

    def _exists(self, key: str):
        """Checks if an analysis is already being computed.
        
        Args:
            key (str): The key of the analysis to check.

        Returns:
            AnalysisFuture: The future that represents the analysis if it is already being computed, None otherwise.
        """

        if self._cache is not None and key in self._cache:
            promise = AnalysisFuture(key)
            promise.set_result(self._cache[key])
            return promise

        if key in self._promises:
            promise = AnalysisFuture(key)
            promise.add_done_callback(self._promise_cancelled)
            self._promises[key].append(promise)
            return promise
        return None

    def _future_done(self, future: Future):
        """Handles the completion of a future.
        
        Args:
            future (Future): The future that completed.

        """

        with self._lock:

            key = self._pending.inverse[future]

            if future.cancelled():
                del self._pending[key]
                del self._promises[key]
                return

            try:
                result = future.result()
                self._cache[key] = result
                error = None
            except (AnalysisError, RuntimeError) as e:
                error = e

            if key not in self._promises:
                return

            if error is None:
                for promise in self._promises[key]:
                    promise.set_result(result)
            else:
                for promise in self._promises[key]:
                    promise.set_exception(error)

            del self._promises[key]
            del self._pending[key]

            with self._wait_condition:
                self._wait_condition.notify()


    def _promise_cancelled(self, future: Future):
        """Handles the cancellation of a promise. If the promise is the last promise for a computation, the computation is cancelled.

        Args:
            future (Future): The promise that was cancelled.

        Returns: 
            bool: True if the promise was the last promise for a computation, False otherwise.

        """

        if not future.cancelled():
            return

        key = future.key

        with self._lock:

            if key not in self._promises:
                return False

            if future not in self._promises[key]:
                return False

            self._promises[key].remove(future)
            if len(self._promises[key]) == 0:
                self._pending[key].cancel()

    @property
    def pending(self):
        """The number of pending analyses."""

        with self._lock:
            return len(self._pending)

    @property
    def total(self):
        """The total number of analyses."""

        with self._lock:
            return self._executor.total

    def cancel(self):
        """Cancels all pending analyses."""

        with self._lock:
            for _, future in list(self._pending.items()):
                future.cancel()

    def wait(self):
        """Waits for all pending analyses to complete. If no analyses are pending, this method returns immediately."""

        if self.total == 0:
            return

        with Progress("Running analysis", self.total) as progress:
            try:

                while True:
                    progress.absolute(self.total - self.pending)
                    if self.pending == 0:
                        break

                    with self._wait_condition:
                        self._wait_condition.wait(1)

            except KeyboardInterrupt:
                self.cancel()
                progress.close()

    def __enter__(self):
        """Sets this analysis processor as the default for the current thread.
        
        Returns:
            AnalysisProcessor: This analysis processor.
        """

        processor = getattr(AnalysisProcessor._context, 'analysis_processor', None)

        if processor == self:
            return self

        if not processor is None:
            logger.warning("Changing default processor for thread %s", threading.current_thread().name)

        AnalysisProcessor._context.analysis_processor = self
        logger.debug("Setting default analysis processor for thread %s", threading.current_thread().name)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Clears the default analysis processor for the current thread."""

        processor = getattr(AnalysisProcessor._context, 'analysis_processor', None)

        if processor == self:
            AnalysisProcessor._context.analysis_processor = None
            self.cancel()

    @staticmethod
    def default():
        """Returns the default analysis processor for the current thread.
        
        Returns: 
            AnalysisProcessor: The default analysis processor for the current thread.
        """

        processor = getattr(AnalysisProcessor._context, 'analysis_processor', None)

        if processor is None:
            logger.warning("Default analysis processor not set for thread %s, using a simple one.", threading.current_thread().name)
            from vot.utilities import ThreadPoolExecutor
            from cachetools import LRUCache
            executor = ThreadPoolExecutor(1)
            cache = LRUCache(1000)
            processor = AnalysisProcessor(executor, cache)
            AnalysisProcessor._context.analysis_processor = processor

        return processor

    @staticmethod
    def commit_default(analysis: Analysis, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence]):
        """Commits an analysis to the default analysis processor. This method is thread-safe. If the analysis is already being computed, this method returns immediately."""
        processor = AnalysisProcessor.default()
        return processor.commit(analysis, experiment, trackers, sequences)

    def run(self, analysis: Analysis, experiment: Experiment,
        trackers: Union[Tracker, List[Tracker]], sequences: Union[Sequence, List[Sequence]]) -> Grid:
        """Runs an analysis on a set of trackers and sequences. This method is thread-safe. If the analysis is already being computed, this method returns immediately.

        Args:
            analysis (Analysis): The analysis to run.
            experiment (Experiment): The experiment to run the analysis on.
            trackers (Union[Tracker, List[Tracker]]): The trackers to run the analysis on.
            sequences (Union[Sequence, List[Sequence]]): The sequences to run the analysis on.

        Returns:
            Grid: The results of the analysis.
        """

        assert self.pending == 0

        future = self.commit(analysis, experiment, trackers, sequences)

        self.wait()

        return future.result()

    @staticmethod
    def run_default(analysis: Analysis, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence]):
        """Runs an analysis on a set of trackers and sequences. This method is thread-safe. If the analysis is already being computed, this method returns immediately.
        
        Args:    
            analysis (Analysis): The analysis to run.
            experiment (Experiment): The experiment to run the analysis on.
            trackers (List[Tracker]): The trackers to run the analysis on.
            sequences (List[Sequence]): The sequences to run the analysis on.
            
        Returns:
            Grid: The results of the analysis."""
        
        processor = AnalysisProcessor.default()
        return processor.run(analysis, experiment, trackers, sequences)


def process_stack_analyses(workspace: "Workspace", trackers: List[Tracker]):
    """Process all analyses in the workspace stack. This function is used by the command line interface to run all the analyses provided in a stack.
    
    Args:
        workspace (Workspace): The workspace to process.
        trackers (List[Tracker]): The trackers to run the analyses on.

    """

    processor = AnalysisProcessor.default()

    results = dict()
    condition = Condition()
    errors = []

    def insert_result(container: dict, key):
        """Creates a callback that inserts the result of a computation into a container. The container is a dictionary that maps analyses to their results.
        
        Args:
            container (dict): The container to insert the result into.
            key (Analysis): The analysis to insert the result for.
        """
        def insert(future: Future):
            """Inserts the result of a computation into a container."""
            try:
                container[key] = future.result()
            except AnalysisError as e:
                errors.append(e)
            except Exception as e:
                logger.exception(e)
            with condition:
                condition.notify()
        return insert

    for experiment in workspace.stack:

        logger.debug("Traversing experiment %s", experiment.identifier)

        experiment_results = dict()

        results[experiment] = experiment_results

        sequences = experiment.transform(workspace.dataset)

        for analysis in experiment.analyses:

            if not analysis.compatible(experiment):
                continue

            logger.debug("Traversing analysis %s", class_fullname(analysis))

            with condition:
                experiment_results[analysis] = None
            promise = processor.commit(analysis, experiment, trackers, sequences)
            promise.add_done_callback(insert_result(experiment_results, analysis))

    if processor.total == 0:
        return results

    logger.debug("Waiting for %d analysis tasks to finish", processor.total)

    with Progress("Running analysis", processor.total) as progress:
        try:

            while True:

                progress.absolute(processor.total - processor.pending)
                if processor.pending == 0:
                    break

                with condition:
                    condition.wait(1)

        except KeyboardInterrupt:
            processor.cancel()
            progress.close()
            logger.info("Analysis interrupted by user, aborting.")
            return None

    if len(errors) > 0:
        logger.info("Errors occured during analysis, incomplete.")
        for e in errors:
            logger.info("Failed task {}: {}".format(e.task, e.root_cause))
            if logger.isEnabledFor(logging.DEBUG):
                e.print(logger)
        return None

    return results