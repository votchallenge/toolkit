import logging
import functools
from enum import Enum, Flag, auto
from typing import List, Optional, Tuple, Dict, Any, Set
from abc import ABC, abstractmethod
from concurrent.futures import Executor

from cachetools import Cache

from vot import VOTException
from vot.tracker import Tracker
from vot.dataset import Sequence
from vot.experiment import Experiment
from vot.region import Region, RegionType
from vot.utilities import class_fullname, arg_hash
from vot.utilities.attributes import Attributee

class MissingResultsException(VOTException):
    pass

class Sorting(Enum):
    UNSORTABLE = auto()
    DESCENDING = auto()
    ASCENDING = auto()

class Hints(Flag):
    NONE = 0
    AXIS_EQUAL = auto()
    SQUARE = auto()

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

    def __init__(self, name: str, abbreviation: Optional[str] = None, hints: Optional[Hints] = Hints.NONE):
        super().__init__(name, abbreviation)
        self._hints = hints

    @property
    def hints(self):
        return self._hints

class Multidimensional(Drawable):
    def __init__(self, name: str, dimensions: int, abbreviation: Optional[str] = None, minimal: Optional[Tuple[float]] = None, \
        maximal: Optional[Tuple[float]] = None, labels: Optional[Tuple[str]] = None, hints: Optional[Hints] = Hints.NONE):
        assert(dimensions > 1)
        super().__init__(name, abbreviation, hints)
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
        maximal: Optional[float] = None, hints: Optional[Hints] = Hints.NONE):
        super().__init__(name, abbreviation, hints)
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

class Table(object):

    def __init__(self, size: tuple):
        self._size = size
        self._data = [None] * functools.reduce(lambda x, y: x * y, size)

    def _ravel(self, pos):
        if not isinstance(pos, tuple):
            pos = (pos, )
        assert(len(pos) == len(self._size))
        raveled = 0
        row = 1
        for n, i in zip(reversed(self._size), reversed(pos)):
            if i < 0 or i >= n:
                raise IndexError("Index out of bounds")
            raveled = i * row + raveled
            row = row * n
        return raveled

    @property
    def dimensions(self):
        return len(self._size)

    def size(self):
        return tuple(self._size)

    def __getitem__(self, i):
        return self._data[self._ravel(i)]

    def __setitem__(self, i, data):
        self._data[self._ravel(i)] = data

class Analysis(Attributee):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._identifier_cache = None

    def compatible(self, experiment: Experiment):
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def identifier(self) -> str:
        if not self._identifier_cache is None:
            return self._identifier_cache

        params = self.dump()
        confighash = arg_hash(**params)

        self._identifier_cache = class_fullname(self) + "@" + confighash

        return self._identifier_cache

    def describe(self) -> Tuple["Result"]:
        """Returns a tuple of descriptions of results
        """
        raise NotImplementedError

    def compute(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence]):
        raise NotImplementedError

    def axes(self):
        """ Returns a tuple of axes of results or None if only a single result tuple is returned """
        raise NotImplementedError

class DependentAnalysis(Analysis):

    @abstractmethod
    def join(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence], results: List[tuple]):
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
    def join(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence], results: List[tuple]):
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
        transformed_results = [[None] * len(sequences) for _ in enumerate(trackers)]
        k = 0
        for i, _ in enumerate(trackers):
            for j, _ in enumerate(sequences):
                transformed_results[i][j] = results[k]
                k += 1
        return transformed_results

    @abstractmethod
    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence):
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
        collapsed = list()
        for tracker, partial in zip(trackers, results):
            collapsed.append(self.collapse(tracker, sequences, partial))
        return collapsed

    @abstractmethod
    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence):
        raise NotImplementedError

    @abstractmethod
    def collapse(self, tracker: Tracker, sequences: List[Sequence], results: List[tuple]):
        raise NotImplementedError

    def axes(self):
        return Axis.TRACKERS,

class TrackerSeparableAnalysis(SeparableAnalysis): # pylint: disable=W0223
    """Separate analysis into multiple per-tracker tasks, each of them is non-separable.
    """

    def join(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence], results: List[tuple]):
        return results

    @abstractmethod
    def subcompute(self, experiment: Experiment, tracker: Tracker, sequences: List[Sequence]):
        raise NotImplementedError

    def separate(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence]):
        parts = []
        for tracker in trackers:
            parts.append((experiment, tracker, sequences))
        return parts

    def axes(self):
        return Axis.TRACKERS,

_ANALYSES = list()

def public(name=None):
    def register(cls):
        _ANALYSES.append(cls)
        return cls
    return register

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


ANALYSIS_PACKAGES = ["vot.analysis.ar", "vot.analysis.eao", "vot.analysis.basic", "vot.analysis.tpr", "vot.analysis.tags"]