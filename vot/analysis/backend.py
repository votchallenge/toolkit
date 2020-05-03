
import traceback
from typing import List, Union
from concurrent.futures import Executor, Future
from threading import RLock

from cachetools import Cache

from vot.dataset import Sequence
from vot.tracker import Tracker
from vot.experiment import Experiment
from vot.analysis import MissingResultsException, SeparatableAnalysis, Analysis, DependentAnalysis
from vot.utilities import arg_hash

def hashkey(analysis, tracker, experiment, sequences):
    if isinstance(sequences, Sequence):
        sequence_hash = sequences.name
    else:
        sequence_hash = arg_hash(*[s.name for s in sequences])
    return (analysis.identifier, tracker.reference, experiment.identifier, sequence_hash)

class AnalysisTask(object):

    def __init__(self, analysis: Analysis, tracker: Tracker, experiment: Experiment, sequences: Union[List[Sequence], Sequence]):
        self._analysis = analysis
        self._tracker = tracker
        self._experiment = experiment
        self._sequences = sequences
        self._key = hashkey(analysis, tracker, experiment, sequences)

    def __call__(self):
        try:
            if isinstance(self._sequences, Sequence):
                result = self._analysis.compute_partial(self._tracker, self._experiment, self._sequences)
            else:
                result = self._analysis.compute(self._tracker, self._experiment, self._sequences)
            return dict(key=self._key, result=result)

        except Exception as e:
            return dict(key=self._key, result=None, exception=e)

    @property
    def key(self):
        return self._key

class AnalysisAggregator(object):
    # TODO: this should be converted into a task

    def __init__(self, analysis: SeparatableAnalysis, count: int, callback):
        self._lock = RLock()
        self._analysis = analysis
        self._count = count
        self._results = [None] * count
        self._callback = callback
        self._tasks = []

    def __call__(self, i, result):
        with self._lock:
            if isinstance(result, Exception):
                self.cancel()
                self._callback(result)
                return
            self._results[i] = result
            self._count = self._count - 1
            if self._count == 0:
                result = self._analysis.join(self._results)
                self._callback(result)

    def append(self, promise):
        self._tasks.append(promise)

    def cancel(self):
        for promise in self._tasks:
            promise.cancel()

class AnalysisProcessor(object):

    def __init__(self, executor: Executor, cache: Cache):
        self._executor = executor
        self._cache = cache
        self._pending = dict()
        self._callbacks = dict()
        self._lock = RLock()

    def submit(self, analysis: Analysis, tracker: Tracker, experiment: Experiment, sequences, callback):

        key = hashkey(analysis, tracker, experiment, sequences)

        with self._lock:

            if self._exists(key, callback):
                return True

            def enumerated_callback(aggregator, i):
                return lambda result: aggregator(i, result)

            if isinstance(analysis, SeparatableAnalysis):

                aggregator = AnalysisAggregator(analysis, len(sequences), lambda x: self._done(key, x))

                for i, sequence in enumerate(sequences):

                    partkey = hashkey(analysis, tracker, experiment, sequence)

                    sequence_callback = enumerated_callback(aggregator, i)

                    if self._exists(partkey, sequence_callback):
                        continue

                    task = AnalysisTask(analysis, tracker, experiment, sequence)
                    promise = self._executor.submit(task)
                    self._callbacks[partkey] = [sequence_callback]
                    promise.add_done_callback(self._future_done)
                    
                self._callbacks[key] = [callback]
                self._pending[key] = aggregator

            if isinstance(analysis, DependentAnalysis):

                dependencies = analysis.dependencies()
                aggregator = AnalysisAggregator(analysis, len(dependencies), lambda x: self._done(key, x))

                for i, sub in enumerate(dependencies):
                    self.submit(sub, tracker, experiment, sequences, enumerated_callback(aggregator, i))

            else:
                task = AnalysisTask(analysis, tracker, experiment, sequences)
                promise = self._executor.submit(task)
                self._callbacks[key] = [callback]
                self._pending[key] = promise
                promise.add_done_callback(self._future_done)

            return False

    def cancel(self, analysis, tracker, experiment, sequences, callback):

        key = hashkey(analysis, tracker, experiment, sequences)

        with self._lock:

            if key not in self._callbacks:
                return False

            self._callbacks[key].remove(callback)
            if len(self._callbacks[key]) == 0:
                self._pending[key].cancel()
                del self._callbacks[key]
                del self._pending[key]
                
    def _exists(self, key, callback):
        if key in self._cache:
            callback(self._cache[key])
            return True

        if key in self._callbacks:
            self._callbacks[key].append(callback)
            return True
        return False

    def _future_done(self, future: Future):

        bundle = future.result()
        key = bundle["key"]

        if not "exception" in bundle:
            result = bundle["result"]
            self._done(key, result)
        else:
            self._done(key, bundle["exception"], False)

    def _done(self, key, result, cache=True):

        with self._lock:

            if cache:
                self._cache[key] = result

            if key not in self._callbacks:
                return

            for callback in self._callbacks[key]:
                callback(result)

            del self._callbacks[key]
            if key in self._pending:
                del self._pending[key]