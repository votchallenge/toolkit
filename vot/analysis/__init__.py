import logging
import functools
import threading
from enum import Enum, Flag, auto
from typing import List, Optional, Tuple, Dict, Any, Set
from abc import ABC, abstractmethod
from concurrent.futures import Executor
import importlib

from cachetools import Cache

from vot import VOTException
from vot.tracker import Tracker
from vot.dataset import Sequence
from vot.experiment import Experiment
from vot.region import Region, RegionType
from vot.utilities import class_fullname, arg_hash
from vot.utilities.data import Grid
from vot.utilities.attributes import Attributee, String

class MissingResultsException(VOTException):
    pass

class Sorting(Enum):
    UNSORTABLE = auto()
    DESCENDING = auto()
    ASCENDING = auto()

class Axis(Enum):
    TRACKERS = auto()
    SEQUENCES = auto()

class Result(ABC):
    """Abstract result object base. This is the base class for all result descriptions.
    """

    def __init__(self, name: str, abbreviation: Optional[str] = None, description: Optional["str"] = ""):
        """Constructor

        Arguments:
            name {str} -- Name of the result, used in reports

        Keyword Arguments:
            abbreviation {Optional[str]} -- Abbreviation, if empty, then name is used. 
            Can be used to define a shorter text representation. (default: {None})
        """
        self._name = name
        if abbreviation is None:
            self._abbreviation = name
        else:
            self._abbreviation = abbreviation

        self._description = description

    @property
    def name(self):
        return self._name

    @property
    def abbreviation(self):
        return self._abbreviation

    @property
    def description(self):
        return self._description

class Label(Result):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Measure(Result):
    """Measure describes a single value numerical output of an analysis. Can have minimum and maximum value as well
    as direction of sorting.
    """

    def __init__(self, name: str, abbreviation: Optional[str] = None, minimal: Optional[float] = None, \
        maximal: Optional[float] = None, direction: Optional[Sorting] = Sorting.UNSORTABLE):
        super().__init__(name, abbreviation)
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

class Drawable(Result):

    def __init__(self, name: str, abbreviation: Optional[str] = None, trait: Optional[str] = None):
        super().__init__(name, abbreviation)
        self._trait = trait

    @property
    def trait(self):
        return self._trait

class Multidimensional(Drawable):
    def __init__(self, name: str, dimensions: int, abbreviation: Optional[str] = None, minimal: Optional[Tuple[float]] = None, \
        maximal: Optional[Tuple[float]] = None, labels: Optional[Tuple[str]] = None, trait: Optional[str] = None):
        assert(dimensions > 1)
        super().__init__(name, abbreviation, trait)
        self._dimensions = dimensions
        self._minimal = minimal
        self._maximal = maximal
        self._labels = labels

    @property
    def dimensions(self):
        return self._dimensions

    def minimal(self, i):
        return self._minimal[i]

    def maximal(self, i):
        return self._maximal[i]

    def label(self, i):
        return self._labels[i]

class Point(Multidimensional):
    """Point is a two or more dimensional numerical output that can be visualized in a scatter plot.
    """

class Plot(Drawable):
    """Plot describes a result in form of a list of values with optional minimum and maximum with respect to some unit. The
    results of the same analysis for different trackers should have the same number of measurements (independent variable).
    """

    def __init__(self, name: str, abbreviation: Optional[str] = None, wrt: str = "frames", minimal: Optional[float] = None, \
        maximal: Optional[float] = None, trait: Optional[str] = None):
        super().__init__(name, abbreviation, trait)
        self._wrt = wrt
        self._minimal = minimal
        self._maximal = maximal

    @property
    def minimal(self):
        return self._minimal

    @property
    def maximal(self):
        return self._maximal


    @property
    def wrt(self):
        return self._wrt

class Curve(Multidimensional):
    """Curve is a list of 2+ dimensional results. The number of elements in a list can vary between samples.
    """

class Analysis(Attributee):

    name = String(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._identifier_cache = None

    def compatible(self, experiment: Experiment):
        raise NotImplementedError

    @property
    def title(self) -> str:
        raise NotImplementedError

    @property
    def identifier(self) -> str:
        if not self._identifier_cache is None:
            return self._identifier_cache

        params = self.dump()
        del params["name"]
        confighash = arg_hash(**params)

        self._identifier_cache = class_fullname(self) + "@" + confighash

        return self._identifier_cache

    def describe(self) -> Tuple["Result"]:
        """Returns a tuple of descriptions of results
        """
        raise NotImplementedError

    def compute(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence]) -> Grid:
        raise NotImplementedError

    def axes(self):
        """ Returns a tuple of axes of results or None if only a single result tuple is returned """
        raise NotImplementedError

    def commit(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence]):
        from vot.analysis.processor import AnalysisProcessor
        return AnalysisProcessor.commit_default(self, experiment, trackers, sequences)

class DependentAnalysis(Analysis):

    @abstractmethod
    def join(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence], results: List[Grid]) -> Grid:
        raise NotImplementedError

    @abstractmethod
    def dependencies(self) -> List[Analysis]:
        raise NotImplementedError

    def compute(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence]):
        partial = []
        for dependency in self.dependencies():
            partial.append(dependency.compute(experiment, trackers, sequences))

        return self.join(experiment, trackers, sequences, partial)

class SeparableAnalysis(Analysis):
    """Base class for all analyses that support separation into multiple sub-tasks.
    """

    @abstractmethod
    def join(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence], results: List[Grid]):
        raise NotImplementedError

    @abstractmethod
    def separate(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence]):
        raise NotImplementedError

    def subcompute(self, *args):
        raise NotImplementedError

    def compute(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence]):
        parts = self.separate(experiment, trackers, sequences)
        results = []
        for part in parts:
            results.append(self.subcompute(*part))
        return self.join(experiment, trackers, sequences, results)

class FullySeparableAnalysis(SeparableAnalysis): # pylint: disable=W0223
    """Analysis that is separable with respect to trackers and sequences, each tracker-sequence pair
    can be its own job.
    """

    def join(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence], results: List[tuple]):
        transformed_results = Grid((len(trackers), len(sequences)))
        k = 0
        for i, _ in enumerate(trackers):
            for j, _ in enumerate(sequences):
                transformed_results[i, j] = results[k]
                k += 1
        return transformed_results

    @abstractmethod
    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence): # pylint: disable=arguments-differ
        raise NotImplementedError

    def separate(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence]):
        parts = []
        for tracker in trackers:
            for sequence in sequences:
                parts.append((experiment, tracker, sequence))
        return parts

    def axes(self):
        return Axis.TRACKERS, Axis.SEQUENCES

class SequenceAveragingAnalysis(FullySeparableAnalysis): # pylint: disable=W0223

    def join(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence], results: List[tuple]):
        results = super().join(experiment, trackers, sequences, results)
        transformed_results = Grid((len(trackers), ))

        for i, tracker in enumerate(trackers):
            transformed_results[i] = self.collapse(tracker, sequences, results.row(i))

        return transformed_results

    @abstractmethod
    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence):
        raise NotImplementedError

    @abstractmethod
    def collapse(self, tracker: Tracker, sequences: List[Sequence], results: Grid):
        raise NotImplementedError

    def axes(self):
        return Axis.TRACKERS,

class SequenceAggregator(DependentAnalysis): # pylint: disable=W0223

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert len(self.dependencies()) == 1 # We only support one dependency for now
        assert self.dependencies()[0].axes() == (Axis.TRACKERS, Axis.SEQUENCES)

    def join(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence], results: List[Grid]):

        results = results[0]
        transformed_results = Grid((len(trackers), ))

        for i, tracker in enumerate(trackers):
            transformed_results[i] = self.aggregate(tracker, sequences, results.row(i))

        return transformed_results

    @abstractmethod
    def aggregate(self, tracker: Tracker, sequences: List[Sequence], results: Grid):
        raise NotImplementedError

    def axes(self):
        return Axis.TRACKERS,

class TrackerSeparableAnalysis(SeparableAnalysis): # pylint: disable=W0223
    """Separate analysis into multiple per-tracker tasks, each of them is non-separable.
    """

    def join(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence], results: List[Grid]):
        transformed_results = Grid((len(trackers), ))
        for i, _ in enumerate(trackers):
            transformed_results[i] = results[i][0, 0]
        return transformed_results

    @abstractmethod
    def subcompute(self, experiment: Experiment, tracker: Tracker, sequences: List[Sequence]):  # pylint: disable=arguments-differ
        raise NotImplementedError

    def separate(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence]):
        parts = []
        for tracker in trackers:
            parts.append((experiment, tracker, sequences))
        return parts

    def axes(self):
        return Axis.TRACKERS,

def simplejoin():
    """Decorator for analyses with join that is simple and can be performed without creating a new task.
    """
    def modify(cls):
        setattr(cls, "simplejoin", True)
        return cls
    return modify

def is_special(region: Region, code=None) -> bool:
    if code is None:
        return region.type == RegionType.SPECIAL
    return region.type == RegionType.SPECIAL and region.code == code

for module in ["vot.analysis.ar", "vot.analysis.eao", "vot.analysis.basic", "vot.analysis.tpr"]:
    importlib.import_module(module)
