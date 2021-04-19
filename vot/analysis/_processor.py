
import logging
import threading
from collections import Iterable, OrderedDict, namedtuple
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
    def transform(arg):
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
    if isinstance(arg, list) and len(arg) == 1:
        return arg[0]
    else:
        return arg

class AnalysisError(ToolkitException):
    def __init__(self, cause, task=None):
        self._tasks = []
        self._cause = cause
        super().__init__(cause, task)
        self._tasks.append(task)

    @property
    def task(self):
        return self._tasks[-1]

    def __str__(self):
        return "Error during analysis {}".format(self.task)

    def print(self, logoutput):
        logoutput.error(str(self))
        if len(self._tasks) > 1:
            for task in reversed(self._tasks[:-1]):
                logoutput.debug("Caused by an error in subtask: %s", str(task))
        logoutput.exception(self.__cause__)

    @property
    def root_cause(self):
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
        promise = Future()
        with self._lock:
            self._queue.put(DebugExecutor.Task(fn, args, kwargs, promise))
            self._semaphor.notify()
            logger.debug("Adding task %s to queue", fn)
            return promise

    def _run(self):

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
                logger.debug("Task %s call resulted in error: %s", task.fn, e)

                error = e

            except Exception as e:

                error = e

                logger.debug("Task %s resulted in exception: %s", task.fn, e)

            if error is not None:
                task.promise.set_exception(error)

                if self._strict:
                    self._alive = False
                    self._clear()
                    break

    def _clear(self):
        with self._lock:

            while True:
                try:
                    task = self._queue.get(False)
                    if not task.promise.done():
                        task.promise.cancel()
                except Empty:
                    break

    def shutdown(self, wait=True):
        self._alive = False
        self._clear()
        if wait:
            with self._lock:
                self._semaphor.notify()

            self._thread.join()

class ExecutorWrapper(object):

    def __init__(self, executor: Executor):
        self._lock = RLock()
        self._executor = executor
        self._pending = OrderedDict()
        self._total = 0

    @property
    def total(self):
        return self._total

    def submit(self, fn, *futures: Tuple[Future], mapping=None) -> Future:

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
        """ Internally handles completion of dependencies
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
            fn (function): [description]
            future (Future): [description]
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
        """

        with self._lock:

            if not future in self._pending:
                return

            dependency = self._pending[future]

            del self._pending[future]

            if future.cancelled():
                dependency.cancel()

class FuturesAggregator(Future):

    def __init__(self, *futures: Tuple[Future]):
        super().__init__()
        self._lock = RLock()
        self._results = [None] * len(futures)
        self._tasks = list(futures)

        for i, future in enumerate(futures):
            future.add_done_callback(partial(self._on_result, i))

        if not self._results:
            self.set_result([])

    def _on_result(self, i, future):
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
        with self._lock:
            try:
                self.set_result(future.result())
            except AnalysisError as e:
                self.set_exception(e)

    def cancel(self):
        with self._lock:
            for promise in self._tasks:
                promise.cancel()
            super().cancel()


class AnalysisTask(object):

    def __init__(self, analysis: Analysis, experiment: Experiment, 
            trackers: List[Tracker], sequences: List[Sequence]):
        self._analysis = analysis
        self._trackers = trackers
        self._experiment = experiment
        self._sequences = sequences
        self._key = hashkey(analysis, experiment, trackers, sequences)

    def __call__(self, dependencies: List[Grid] = None):
        try:
            if dependencies is None:
                dependencies = []
            return self._analysis.compute(self._experiment, self._trackers, self._sequences, dependencies)
        except BaseException as e:
            raise AnalysisError(cause=e, task=self._key)

class AnalysisPartTask(object):

    def __init__(self, analysis: SeparableAnalysis, experiment: Experiment,
            trackers: List[Tracker], sequences: List[Sequence]):
        self._analysis = analysis
        self._trackers = trackers
        self._experiment = experiment
        self._sequences = sequences
        self._key = hashkey(analysis, experiment, unwrap(trackers), unwrap(sequences))

    def __call__(self, dependencies: List[Grid] = None):
        try:
            if dependencies is None:
                dependencies = []
            return self._analysis.compute(self._experiment, self._trackers, self._sequences, dependencies)
        except BaseException as e:
            raise AnalysisError(cause=e, task=self._key)

class AnalysisJoinTask(object):

    def __init__(self, analysis: SeparableAnalysis, experiment: Experiment,
            trackers: List[Tracker], sequences: List[Sequence]):
        self._analysis = analysis
        self._trackers = trackers
        self._experiment = experiment
        self._sequences = sequences
        self._key = hashkey(analysis, experiment, trackers, sequences)

    def __call__(self, results: List[Grid]):
        try:
            return self._analysis.join(self._trackers, self._sequences, results)
        except BaseException as e:
            raise AnalysisError(cause=e, task=self._key)

class AnalysisFuture(Future):

    def __init__(self, key):
        super().__init__()
        self._key = key

    @property
    def key(self):
        return self._key

    def __repr__(self) -> str:
        return "<AnalysisFuture key={}>".format(self._key)

class AnalysisProcessor(object):

    _context = threading.local()

    def __init__(self, executor: Executor = None, cache: Cache = None):
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

    def _exists(self, key):
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
        with self._lock:
            return len(self._pending)

    @property
    def total(self):
        with self._lock:
            return self._executor.total

    def cancel(self):
        with self._lock:
            for _, future in list(self._pending.items()):
                future.cancel()

    def wait(self):

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

        processor = getattr(AnalysisProcessor._context, 'analysis_processor', None)

        if processor == self:
            return self

        if not processor is None:
            logger.warning("Changing default processor for thread %s", threading.current_thread().name)

        AnalysisProcessor._context.analysis_processor = self
        logger.debug("Setting default analysis processor for thread %s", threading.current_thread().name)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        processor = getattr(AnalysisProcessor._context, 'analysis_processor', None)

        if processor == self:
            AnalysisProcessor._context.analysis_processor = None
            self.cancel()

    @staticmethod
    def default():

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
        processor = AnalysisProcessor.default()
        return processor.commit(analysis, experiment, trackers, sequences)

    def run(self, analysis: Analysis, experiment: Experiment,
        trackers: Union[Tracker, List[Tracker]], sequences: Union[Sequence, List[Sequence]]) -> Grid:

        assert self.pending == 0

        future = self.commit(analysis, experiment, trackers, sequences)

        self.wait()

        return future.result()

    @staticmethod
    def run_default(analysis: Analysis, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence]):
        processor = AnalysisProcessor.default()
        return processor.run(analysis, experiment, trackers, sequences)


def process_stack_analyses(workspace: "Workspace", trackers: List[Tracker]):

    processor = AnalysisProcessor.default()

    results = dict()
    condition = Condition()

    def insert_result(container: dict, key):
        def insert(future: Future):
            try:
                container[key] = future.result()
            except AnalysisError as e:
                e.print(logger)
            except Exception as e:
                logger.exception(e)
            with condition:
                condition.notify()
        return insert

    for experiment in workspace.stack:

        logger.debug("Traversing experiment %s", experiment.identifier)

        experiment_results = dict()

        results[experiment] = experiment_results

        sequences = [experiment.transform(sequence) for sequence in workspace.dataset]

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

    return results