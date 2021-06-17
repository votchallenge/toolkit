import logging
import functools
import threading
from collections import namedtuple
from enum import Enum, Flag, auto
from typing import List, Optional, Tuple, Dict, Any, Set, Union, NamedTuple
from abc import ABC, abstractmethod
from concurrent.futures import Executor
import importlib

from cachetools import Cache
from class_registry import ClassRegistry

from attributee import Attributee, String

from vot import ToolkitException
from vot.tracker import Tracker
from vot.dataset import Sequence
from vot.experiment import Experiment
from vot.region import Region, RegionType
from vot.utilities import class_fullname, arg_hash
from vot.utilities.data import Grid

analysis_registry = ClassRegistry("vot_analysis")

class MissingResultsException(ToolkitException):
    """Exception class that denotes missing results during analysis
    """
    pass

class Sorting(Enum):
    """Sorting direction enumeration class
    """
    UNSORTABLE = auto()
    DESCENDING = auto()
    ASCENDING = auto()

class Axes(Enum):
    """Semantic information for axis in analysis grid
    """
    NONE = auto()
    TRACKERS = auto()
    SEQUENCES = auto()
    BOTH = auto()

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
    def name(self) -> str:
        return self._name

    @property
    def abbreviation(self) -> str:
        return self._abbreviation

    @property
    def description(self) -> str:
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
    def minimal(self) -> float:
        return self._minimal

    @property
    def maximal(self) -> float:
        return self._maximal

    @property
    def direction(self) -> Sorting:
        return self._direction

class Drawable(Result):
    """Base class for results that can be visualized in plots.
    """

    def __init__(self, name: str, abbreviation: Optional[str] = None, trait: Optional[str] = None):
        """[summary]

        Args:
            name (str): [description]
            abbreviation (Optional[str], optional): [description]. Defaults to None.
            trait (Optional[str], optional): Trait of the data, used for specification . Defaults to None.
        """
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
        raise NotImplementedError()

    @property
    def title(self) -> str:
        raise NotImplementedError()

    def dependencies(self) -> List["Analysis"]:
        return []

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
        raise NotImplementedError()

    def compute(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence], dependencies: List[Grid]) -> Grid:
        raise NotImplementedError()

    @property
    def axes(self) -> Axes:
        """ Returns axes semantic description for the result grid """
        raise NotImplementedError()

    def commit(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence]):
        return AnalysisProcessor.commit_default(self, experiment, trackers, sequences)

    def run(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence]):
        return AnalysisProcessor.run_default(self, experiment, trackers, sequences)

class SeparableAnalysis(Analysis):
    """Analysis that is separable with respect to trackers and/or sequences, each part can be processed in parallel
    as a separate job. The separation is determined by the result of the axes() method: Axes.BOTH means separation
    in tracker-sequence pairs, Axes.TRACKER means separation according to  
    """

    SeparablePart = namedtuple("SeparablePart", ["trackers", "sequences", "tid", "sid"])

    @abstractmethod
    def subcompute(self, experiment: Experiment, tracker, sequence, dependencies: List[Grid]) -> Tuple[Any]:
        """This method is called for every part of the analysis.

        Args:
            experiment (Experiment): [description]
            tracker ([type]): [description]
            sequence ([type]): [description]
            dependencies (List[Grid]): Dependencies of the analysis, 
                note that each dependency is processed using select function to only contain 
                information relevant for the current part of the analysis

        Raises:
            NotImplementedError: [description]

        Returns:
            Tuple[Any]: [description]
        """
        raise NotImplementedError()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # All dependencies should be mappable to individual parts. If parts contain 
        # separation only across trackers or sequences then we are unable to properly
        # assign dependencies that contain individual 
        if self.axes != Axes.BOTH:
            assert all([dependency.axes != Axes.BOTH for dependency in self.dependencies()])

    def separate(self, trackers: List[Tracker], sequences: List[Sequence]) -> List["SeparablePart"]:
        if self.axes == Axes.BOTH:
            parts = []
            for i, tracker in enumerate(trackers):
                for j, sequence in enumerate(sequences):
                    parts.append(SeparableAnalysis.SeparablePart([tracker], [sequence], i, j))
            return parts
        elif self.axes == Axes.TRACKERS:
            parts = []
            for i, tracker in enumerate(trackers):
                parts.append(SeparableAnalysis.SeparablePart([tracker], sequences, i, None))
            return parts
        elif self.axes == Axes.SEQUENCES:
            parts = []
            for j, sequence in enumerate(sequences):
                parts.append(SeparableAnalysis.SeparablePart(trackers, [sequence], None, j))
            return parts

    def join(self, trackers: List[Tracker], sequences: List[Sequence], results: List[Tuple[Any]]):
        if self.axes == Axes.BOTH:
            transformed_results = Grid(len(trackers), len(sequences))
            k = 0
            for i, _ in enumerate(trackers):
                for j, _ in enumerate(sequences):
                    transformed_results[i, j] = results[k][0,0]
                    k += 1
            return transformed_results
        elif self.axes == Axes.TRACKERS:
            transformed_results = Grid(len(trackers), 1)
            k = 0
            for i, _ in enumerate(trackers):
                transformed_results[i, 0] = results[k][0,0]
                k += 1
            return transformed_results
        elif self.axes == Axes.SEQUENCES:
            transformed_results = Grid(1, len(sequences))
            k = 0
            for i, _ in enumerate(sequences):
                transformed_results[0, i] = results[k][0,0]
                k += 1
            return transformed_results

    @staticmethod
    def select(meta: Analysis, data: Grid, tracker: int, sequence: int) -> Grid:
        """Select appropriate subpart of dependency results for the part, used internally by sequential and
        parallel processor. This method handles propagation across "singleton" dimension. 
        
        The idea is that a certain part of the analysis will only require the part of the result corresponding
        to the tracker and/or sequence that it is processing.

        Args:
            meta (Analysis): Description of the dependency analysis
            data (Grid): Returned data of the dependency
            tracker (int): Index of the tracker required by the part or None
            sequence (int): Index of the sequence required by the part or None

        Returns:
            Grid: Subsection of the result, still in Grid format.
        """
        if meta.axes == Axes.BOTH:
            return data.cell(tracker, sequence)
        elif meta.axes == Axes.TRACKERS:
            return data.row(tracker)
        elif meta.axes == Axes.SEQUENCES:
            return data.column(sequence)
        else:
            return data

    def compute(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence], dependencies: List[Grid]) -> Grid:
        """The blocking non-parallel version of computation that can be called directly. Splits the job in parts and
        runs them sequentially. For parallel execution use the analysis processor.

        Args:
            experiment (Experiment): Experiment from which to take results
            trackers (List[Tracker]): Trackers to run analysis on
            sequences (List[Sequence]): Sequences to run analysis on
            dependencies (List[Grid]): Results from depndencies, if you override the class and add dependencies, you also
            have to override this function and handle them.

        Returns:
            Grid: Results in a data grid object
        """

        if self.axes == Axes.BOTH and len(trackers) == 1 and len(sequences) == 1:
            return Grid.scalar(self.subcompute(experiment, trackers[0], sequences[0], dependencies))
        elif self.axes == Axes.TRACKERS and len(trackers) == 1:
            return Grid.scalar(self.subcompute(experiment, trackers[0], sequences, dependencies))
        elif self.axes == Axes.SEQUENCES and len(sequences) == 1:
            return Grid.scalar(self.subcompute(experiment, trackers, sequences[0], dependencies))
        else:
            parts = self.separate(trackers, sequences)
            results = []
            for part in parts:
                partdependencies = [SeparableAnalysis.select(meta, data, part.tid, part.sid) 
                    for meta, data in zip(self.dependencies(), dependencies)]
                results.append(self.compute(experiment, part.trackers, part.sequences, partdependencies))

            return self.join(trackers, sequences, results)

    @property
    def axes(self) -> Axes:
        return Axes.BOTH

class SequenceAggregator(Analysis): # pylint: disable=W0223

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # We only support one dependency in aggregator ...
        assert len(self.dependencies()) == 1
        # ... it should produce a grid of results that can be averaged over sequences
        assert self.dependencies()[0].axes == Axes.BOTH

    @abstractmethod
    def aggregate(self, tracker: Tracker, sequences: List[Sequence], results: Grid) -> Tuple[Any]:
        raise NotImplementedError()

    def compute(self, _: Experiment, trackers: List[Tracker], sequences: List[Sequence], dependencies: List[Grid]) -> Grid:
        results = dependencies[0]
        transformed_results = Grid(len(trackers), 1)

        for i, tracker in enumerate(trackers):
            transformed_results[i, 0] = self.aggregate(tracker, sequences, results.row(i))

        return transformed_results

    @property
    def axes(self) -> Axes:
        return Axes.TRACKERS

class TrackerSeparableAnalysis(SeparableAnalysis):
    """Separate analysis into multiple per-tracker tasks, each of them is non-separable.
    """

    @abstractmethod
    def subcompute(self, experiment: Experiment, tracker: Tracker, sequences: List[Sequence], dependencies: List[Grid]) -> Tuple[Any]:
        raise NotImplementedError()

    @property
    def axes(self) -> Axes:
        return Axes.TRACKERS

class SequenceSeparableAnalysis(SeparableAnalysis):
    """Separate analysis into multiple per-tracker tasks, each of them is non-separable.
    """

    @abstractmethod
    def subcompute(self, experiment: Experiment, trackers: List[Tracker], sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:
        raise NotImplementedError

    @property
    def axes(self) -> Axes:
        return Axes.SEQUENCES

def is_special(region: Region, code=None) -> bool:
    if code is None:
        return region.type == RegionType.SPECIAL
    return region.type == RegionType.SPECIAL and region.code == code

from ._processor import process_stack_analyses, AnalysisProcessor, AnalysisError

for module in ["vot.analysis.multistart", "vot.analysis.supervised", "vot.analysis.basic", "vot.analysis.tpr"]:
    importlib.import_module(module)
