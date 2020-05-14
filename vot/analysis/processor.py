
import logging
from collections import Iterable
from typing import List, Union, Callable
from concurrent.futures import Executor, Future
from threading import RLock

from cachetools import Cache
from bidict import bidict

from vot import VOTException
from vot.dataset import Sequence
from vot.tracker import Tracker
from vot.experiment import Experiment
from vot.analysis import SeparableAnalysis, Analysis, DependentAnalysis
from vot.utilities import arg_hash

logger = logging.getLogger("vot")

def hashkey(analysis: Analysis, *args):
    def transform(arg):
        if isinstance(arg, Sequence):
            return arg.name
        if isinstance(arg, Tracker):
            return arg.reference
        if isinstance(arg, Experiment):
            return arg.identifier
        if isinstance(arg, Iterable):
            return arg_hash(*[transform(i) for i in arg])

    return (analysis.identifier, *[transform(arg) for arg in args])

class AnalysisError(VOTException):
    def __init__(self, cause, task):
        self._tasks = []
        if isinstance(cause, AnalysisError):
            self._tasks.extend(cause._tasks)
            cause = cause.__cause__ or cause.__context__
        super().__init__(cause)
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

class AnalysisTask(object):

    def __init__(self, analysis: Analysis, experiment: Experiment, 
            trackers: List[Tracker], sequences: List[Sequence]):
        self._analysis = analysis
        self._trackers = trackers
        self._experiment = experiment
        self._sequences = sequences
        self._key = hashkey(analysis, experiment, trackers, sequences)

    def __call__(self):
        try:
            return self._analysis.compute(self._experiment, self._trackers, self._sequences)
        except BaseException as e:
            raise AnalysisError(e, task=self._key)

class AnalysisPartTask(object):

    def __init__(self, analysis: SeparableAnalysis, arguments: tuple):
        self._analysis = analysis
        self._arguments = arguments
        self._key = hashkey(analysis, *arguments)

    def __call__(self):
        try:
            return self._analysis.subcompute(*self._arguments)
        except BaseException as e:
            raise AnalysisError(e, task=self._key)

class AnalysisJoinTask(object):

    def __init__(self, analysis: Union[SeparableAnalysis, DependentAnalysis], experiment: Experiment,
            trackers: List[Tracker], sequences: List[Sequence], results):
        self._analysis = analysis
        self._trackers = trackers
        self._experiment = experiment
        self._sequences = sequences
        self._results = results
        self._key = hashkey(analysis, experiment, trackers, sequences)

    def __call__(self):
        try:
            return self._analysis.join(self._experiment, self._trackers, self._sequences, self._results)
        except BaseException as e:
            raise AnalysisError(e, task=self._key)

class AnalysisAggregator(Future):

    def __init__(self, key: tuple, analysis: Union[SeparableAnalysis, DependentAnalysis],
        experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence], count: int, executor):
        super().__init__()
        self._lock = RLock()
        self._analysis = analysis
        self._trackers = trackers
        self._experiment = experiment
        self._sequences = sequences
        self._count = count
        self._results = [None] * count
        self._executor = executor
        self._tasks = []
        self._key = key

    def _on_result(self, result, i):
        with self._lock:
            if self.done():
                return
            if isinstance(result, Exception):
                for promise in self._tasks:
                    promise.cancel()
                self.set_exception(AnalysisError(result, task=self._key))
                return
            
            self._results[i] = result
            self._count = self._count - 1
            if self._count == 0:
                promise = self._executor.submit(AnalysisJoinTask(self._analysis, self._experiment,
                    self._trackers, self._sequences, self._results))
                promise.add_done_callback(self._on_done)

    def _on_done(self, future):
        with self._lock:
            try:
                self.set_result(future.result())
            except AnalysisError as e:
                self.set_exception(e)

    def callback(self, i):
        return lambda result: self._on_result(result, i)

    def append(self, promise):
        self._tasks.append(promise)

    def cancel(self):
        with self._lock:
            for promise in self._tasks:
                promise.cancel()
            super().cancel()

class AnalysisProcessor(object):

    def __init__(self, executor: Executor, cache: Cache):
        self._executor = executor
        self._cache = cache
        self._pending = bidict()
        self._callbacks = dict()
        self._lock = RLock()
        self._total = 0

    def submit(self, analysis: Analysis, experiment: Experiment,
        trackers: Union[Tracker, List[Tracker]], sequences: Union[Sequence, List], callback: Callable):

        key = hashkey(analysis, experiment, trackers, sequences)

        with self._lock:

            if self._exists(key, callback):
                return True
    
            if isinstance(analysis, SeparableAnalysis):

                parts = analysis.separate(experiment, trackers, sequences)
                aggregator = AnalysisAggregator(key, analysis, experiment, trackers, sequences, len(parts), self._executor)
                aggregator.add_done_callback(self._future_done)
                self._callbacks[key] = [callback]
                self._pending[key] = aggregator
                self._total += 1

                for i, part in enumerate(parts):
                    partkey = hashkey(analysis, *part)

                    sequence_callback = aggregator.callback(i)

                    if self._exists(partkey, sequence_callback):
                        continue

                    task = AnalysisPartTask(analysis, part)
                    promise = self._executor.submit(task)
                    self._callbacks[partkey] = [sequence_callback]
                    self._pending[partkey] = promise
                    self._total += 1
                    aggregator.append(promise)
                    promise.add_done_callback(self._future_done)
    

            elif isinstance(analysis, DependentAnalysis):

                dependencies = analysis.dependencies()
                aggregator = AnalysisAggregator(key, analysis, experiment,
                        trackers, sequences, len(dependencies), self._executor)
                aggregator.add_done_callback(self._future_done)

                self._callbacks[key] = [callback]
                self._pending[key] = aggregator
                self._total += 1

                for i, sub in enumerate(dependencies):
                    self.submit(sub, experiment, trackers, sequences, aggregator.callback(i))

            else:
                task = AnalysisTask(analysis, experiment, trackers, sequences)
                promise = self._executor.submit(task)
                self._callbacks[key] = [callback]
                self._pending[key] = promise
                self._total += 1
                promise.add_done_callback(self._future_done)
                logger.debug("Adding analysis task %s", key)
            
            return False

    def cancel(self, analysis, experiment, trackers, sequences, callback):

        key = hashkey(analysis, experiment, trackers, sequences)

        with self._lock:

            if key not in self._callbacks:
                return False

            if callback not in self._callbacks[key]:
                return False

            self._callbacks[key].remove(callback)
            if len(self._callbacks[key]) == 0:
                self._pending[key].cancel()
                
    def _exists(self, key, callback):
        if key in self._cache:
            callback(self._cache[key])
            return True

        if key in self._callbacks:
            self._callbacks[key].append(callback)
            return True
        return False

    def _future_done(self, future: Future):

        with self._lock:

            key = self._pending.inverse[future]

            if future.cancelled():
                del self._pending[key]
                del self._callbacks[key]
                return
            try:
                result = future.result()
                self._done(key, result, True)
            except AnalysisError as e:
                self._done(e.task, e, False)

    def _done(self, key, result, cache=True):

        if cache:
            self._cache[key] = result

        if key not in self._callbacks:
            return

        for callback in self._callbacks[key]:
            callback(result)

        del self._callbacks[key]
        del self._pending[key]

    @property
    def pending(self):
        with self._lock:
            return len(self._pending)

    @property
    def total(self):
        with self._lock:
            return self._total

    def cancel_all(self):
        with self._lock:
            for _, future in list(self._pending.items()):
                future.cancel()

def process_stack_analyses(workspace: "Workspace", trackers: List[Tracker], executor: Executor, cache: Cache):

    from vot.utilities import Progress
    from threading import Condition

    processor = AnalysisProcessor(executor, cache)

    results = dict()
    condition = Condition()

    def insert_result(container: dict, key):
        def insert(x):
            if isinstance(x, Exception):
                if isinstance(x, AnalysisError):
                    x.print(logger)
                else:
                    logger.exception(x)
            else:
                container[key] = x
            with condition:
                condition.notify()
        return insert

    for experiment in workspace.stack:

        logger.debug("Traversing experiment %s", experiment.identifier)

        experiment_results = dict()

        results[experiment] = experiment_results

        for analysis in workspace.stack.analyses(experiment):

            if not analysis.compatible(experiment):
                continue

            logger.debug("Traversing analysis %s", analysis.name)

            with condition:
                experiment_results[analysis] = None
            processor.submit(analysis, experiment, trackers, workspace.dataset, insert_result(experiment_results, analysis))

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
            processor.cancel_all()
            progress.close()
            logger.info("Analysis interrupted by user, aborting.")
            return None

    return results