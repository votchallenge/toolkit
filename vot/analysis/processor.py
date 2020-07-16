
import logging
import threading
from collections import Iterable
from typing import List, Union
from concurrent.futures import Executor, Future
from threading import RLock, Condition

from cachetools import Cache
from bidict import bidict

from vot import VOTException
from vot.dataset import Sequence
from vot.tracker import Tracker
from vot.experiment import Experiment
from vot.analysis import SeparableAnalysis, Analysis, DependentAnalysis
from vot.utilities import arg_hash, class_fullname, Progress

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

    def _on_result(self, future, i):
        with self._lock:
            if self.done():
                return
            try:
                self._results[i] = future.result()
            except AnalysisError as e:
                for promise in self._tasks:
                    promise.cancel()
                self.set_exception(AnalysisError(e, task=self._key))

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
        return lambda future: self._on_result(future, i)

    def append(self, promise):
        self._tasks.append(promise)

    def cancel(self):
        with self._lock:
            for promise in self._tasks:
                promise.cancel()
            super().cancel()


class AnalysisFuture(Future):

    def __init__(self, key):
        super().__init__()
        self._key = key

    @property
    def key(self):
        return self._key

class AnalysisProcessor(object):

    _context = threading.local()

    def __init__(self, executor: Executor, cache: Cache):
        self._executor = executor
        self._cache = cache
        self._pending = bidict()
        self._promises = dict()
        self._lock = RLock()
        self._total = 0
        self._wait_condition = Condition()

    def commit(self, analysis: Analysis, experiment: Experiment,
        trackers: Union[Tracker, List[Tracker]], sequences: Union[Sequence, List]):

        key = hashkey(analysis, experiment, trackers, sequences)

        with self._lock:

            promise = self._exists(key)

            if not promise is None:
                return promise
    
            promise = AnalysisFuture(key)
            promise.add_done_callback(self._promise_cancelled)

            if isinstance(analysis, SeparableAnalysis):

                parts = analysis.separate(experiment, trackers, sequences)
                aggregator = AnalysisAggregator(key, analysis, experiment, trackers, sequences, len(parts), self._executor)
                aggregator.add_done_callback(self._future_done)
                self._promises[key] = [promise]
                self._pending[key] = aggregator
                self._total += 1

                for i, part in enumerate(parts):
                    partkey = hashkey(analysis, *part)

                    sequence_callback = aggregator.callback(i)

                    partpromise = self._exists(partkey)
                    if not partpromise is None:
                        partpromise.add_done_callback(sequence_callback)
                        continue

                    partpromise = AnalysisFuture(key)
                    partpromise.add_done_callback(sequence_callback)

                    task = AnalysisPartTask(analysis, part)
                    executorpromise = self._executor.submit(task)
                    self._promises[partkey] = [partpromise]
                    self._pending[partkey] = executorpromise
                    self._total += 1
                    aggregator.append(executorpromise)
                    executorpromise.add_done_callback(self._future_done)
    

            elif isinstance(analysis, DependentAnalysis):

                dependencies = analysis.dependencies()
                aggregator = AnalysisAggregator(key, analysis, experiment,
                        trackers, sequences, len(dependencies), self._executor)
                aggregator.add_done_callback(self._future_done)

                self._promises[key] = [promise]
                self._pending[key] = aggregator
                self._total += 1

                for i, sub in enumerate(dependencies):
                    subpromise = self.commit(sub, experiment, trackers, sequences)
                    subpromise.add_done_callback(aggregator.callback(i))

            else:
                task = AnalysisTask(analysis, experiment, trackers, sequences)
                executorpromise = self._executor.submit(task)
                promise = AnalysisFuture(key)
                self._promises[key] = [promise]
                self._pending[key] = executorpromise
                self._total += 1
                executorpromise.add_done_callback(self._future_done)
                logger.debug("Adding analysis task %s", key)
            
            return promise

    def _exists(self, key):
        if key in self._cache:
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
            except AnalysisError as e:
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
            return self._total

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
    def commit_default(analysis: AnalysisFuture, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence]):
        processor = AnalysisProcessor.default()
        return processor.commit(analysis, experiment, trackers, sequences)

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