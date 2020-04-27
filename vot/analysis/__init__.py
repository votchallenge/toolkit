from typing import List, Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod

from vot.tracker import Tracker
from vot.dataset import Sequence
from vot.experiment import Experiment
from vot.region import Region, RegionType
from vot.utilities import class_fullname, arg_hash

class MissingResultsException(Exception):
    pass

def is_special(region: Region, code = None) -> bool:
    if code is None:
        return region.type == RegionType.SPECIAL
    return region.type == RegionType.SPECIAL and region.code == code

class Analysis(ABC):

    def __init__(self):
        self._identifier_cache = None

    def compatible(self, experiment: Experiment):
        return False

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    def identifier(self) -> str:
        if not self._identifier_cache is None:
            return self._identifier_cache

        params = self.parameters()
        confighash = arg_hash(**params)

        self._identifier_cache = class_fullname(self) + "@" + confighash

        return self._identifier_cache

    def parameters(self) -> Dict[str, Any]:
        return dict()

    @abstractmethod
    def describe(self) -> Tuple["Measure"]:
        raise NotImplementedError

    @abstractmethod
    def compute(self, tracker: Tracker, experiment: Experiment, sequences: List[Sequence]):
        raise NotImplementedError

class Result(ABC):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

class Measure(Result):

    DESCENDING = "descending"
    ASCENDING = "ascending"

    def __init__(self, name: str, minimal: Optional[float] = None, \
        maximal: Optional[float] = None, direction: Optional[str] = ASCENDING):
        super().__init__(name)
        self._minimal = minimal
        self._maximal = maximal
        self._direction = direction

    @property
    def minimal(self):
        return self._minimal

    @property
    def maximal(self):
        return self._maximal

    @property
    def direction(self):
        return self._direction

class Curve(Result):

    def __init__(self, name: str):
        super().__init__(name)


class SeparatableAnalysis(Analysis):

    @abstractmethod
    def join(self, results: List[tuple]):
        raise NotImplementedError

    @abstractmethod
    def compute_partial(self, tracker: Tracker, experiment: Experiment, sequence: Sequence):
        raise NotImplementedError

    def compute(self, tracker: Tracker, experiment: Experiment, sequences: List[Sequence]):
        partial = []
        for sequence in sequences:
            partial.append(self.compute_partial(tracker, experiment, sequence))

        return self.join(partial)

class DependentAnalysis(Analysis):

    @abstractmethod
    def join(self, results: List[tuple]):
        raise NotImplementedError

    @abstractmethod
    def dependencies(self) -> List[Analysis]:
        raise NotImplementedError

    def compute(self, tracker: Tracker, experiment: Experiment, sequences: List[Sequence]):
        partial = []
        for dependency in self.dependencies():
            partial.append(dependency.compute(tracker, experiment, sequences))

        return self.join(partial)

_ANALYSES = list()

def register_analysis(analysis: Analysis):
    _ANALYSES.append(analysis)

def process_measures(workspace: "Workspace", trackers: List[Tracker]):

    results = dict()

    for experiment in workspace.stack:

        results[experiment.identifier] = list()

        for tracker in trackers:

            tracker_results = {}
            tracker_results['tracker_name'] = tracker.identifier

            for analysis in workspace.stack.analyses(experiment):

                if not analysis.compatible(experiment):
                    continue

                tracker_results[class_fullname(analysis)] = analysis.compute(tracker, experiment, workspace.dataset)

            results[experiment.identifier].append(tracker_results)

    return results