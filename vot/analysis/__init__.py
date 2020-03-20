from typing import List, Optional
from abc import ABC, abstractmethod

from vot.tracker import Tracker
from vot.dataset import Sequence
from vot.experiment import Experiment
from vot.region import Region, RegionType
from vot.utilities import class_fullname

class MissingResultsException(Exception):
    pass

def is_special(region: Region, code = None) -> bool:
    if code is None:
        return region.type == RegionType.SPECIAL
    return region.type == RegionType.SPECIAL and region.code == code

class PerformanceMeasure(ABC):

    def compatible(self, experiment: Experiment):
        return False

    @abstractmethod
    def compute(self, tracker: Tracker, experiment: Experiment):
        raise NotImplementedError

class MeasureDescription(object):

    DESCENDING = "descending"
    ASCENDING = "ascending"

    def __init__(self, name: str, minimal: Optional[float], \
        maximal: Optional[float], direction: Optional[str]):
        self._name = name
        self._minimal = minimal
        self._maximal = maximal
        self._direction = direction

    @property
    def name(self):
        return self._name

    @property
    def minimal(self):
        return self._minimal

    @property
    def maximal(self):
        return self._maximal

    @property
    def direction(self):
        return self._direction

class SeparatablePerformanceMeasure(PerformanceMeasure):

    @abstractmethod
    def join(self, results: List[tuple]):
        raise NotImplementedError

    @abstractmethod
    def compute_partial(self, tracker: Tracker, experiment: Experiment, sequence: Sequence):
        raise NotImplementedError

    def compute(self, tracker: Tracker, experiment: Experiment):
        partial = []
        for sequence in experiment.workspace.dataset:
            partial.append(self.compute_partial(tracker, experiment, sequence))

        return self.join(partial)

class NonSeparatablePerformanceMeasure(PerformanceMeasure):

    @abstractmethod
    def compute_measure(self, tracker: Tracker, experiment: Experiment):
        raise NotImplementedError

    def compute(self, tracker: Tracker, experiment: Experiment):
        return self.compute_measure(tracker, experiment)

_MEASURES = list()

def register_measure(measure: PerformanceMeasure):
    _MEASURES.append(measure)


def process_measures(workspace: "Workspace", trackers: List[Tracker]):

    results = dict()

    for experiment in workspace.stack:

        results[experiment.identifier] = list()

        for tracker in trackers:

            tracker_results = {}
            tracker_results['tracker_name'] = tracker.identifier

            for measure in workspace.stack.measures(experiment):

                if not measure.compatible(experiment):
                    continue

                tracker_results[class_fullname(measure)] = measure.compute(tracker, experiment)

            results[experiment.identifier].append(tracker_results)

    return results