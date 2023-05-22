"""This module contains classes and functions for analysis of tracker performance. The analysis is performed on the results of an experiment."""

from collections import namedtuple
from enum import Enum, auto
from typing import List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import importlib

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
    def __init__(self, *args: object) -> None:
        """Constructor"""
        if not args:
            args = ["Missing results"]
        super().__init__(*args)
        
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
        """Name of the result, used in reports"""
        return self._name

    @property
    def abbreviation(self) -> str:
        """Abbreviation, if empty, then name is used. Can be used to define a shorter text representation."""
        return self._abbreviation

    @property
    def description(self) -> str:
        """Description of the result, used in reports"""
        return self._description

class Label(Result):
    """Label describes a single categorical output of an analysis. Can have a set of possible values."""

    def __init__(self, *args, **kwargs):
        """Constructor."""
        super().__init__(*args, **kwargs)

class Measure(Result):
    """Measure describes a single value numerical output of an analysis. Can have minimum and maximum value as well
    as direction of sorting.
    """

    def __init__(self, name: str, abbreviation: Optional[str] = None, minimal: Optional[float] = None, \
        maximal: Optional[float] = None, direction: Optional[Sorting] = Sorting.UNSORTABLE):
        """Constructor for Measure class.
        
        Arguments:
            name {str} -- Name of the measure, used in reports

        Keyword Arguments:
            abbreviation {Optional[str]} -- Abbreviation, if empty, then name is used.
            Can be used to define a shorter text representation. (default: {None})
            minimal {Optional[float]} -- Minimal value of the measure. If None, then the measure is not bounded from below. (default: {None})
            maximal {Optional[float]} -- Maximal value of the measure. If None, then the measure is not bounded from above. (default: {None})
            direction {Optional[Sorting]} -- Direction of sorting. If Sorting.UNSORTABLE, then the measure is not sortable. (default: {Sorting.UNSORTABLE})

        """

        super().__init__(name, abbreviation)
        self._minimal = minimal
        self._maximal = maximal
        self._direction = direction

    @property
    def minimal(self) -> float:
        """Minimal value of the measure. If None, then the measure is not bounded from below."""
        return self._minimal

    @property
    def maximal(self) -> float:
        """Maximal value of the measure. If None, then the measure is not bounded from above."""
        return self._maximal

    @property
    def direction(self) -> Sorting:
        """Direction of sorting. If Sorting.UNSORTABLE, then the measure is not sortable."""
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
        """Trait of the data, used for specification"""
        return self._trait

class Multidimensional(Drawable):
    """Base class for multidimensional results. This class is used to describe results that can be visualized in a scatter plot."""

    def __init__(self, name: str, dimensions: int, abbreviation: Optional[str] = None, minimal: Optional[Tuple[float]] = None, \
        maximal: Optional[Tuple[float]] = None, labels: Optional[Tuple[str]] = None, trait: Optional[str] = None):
        """Constructor for Multidimensional class.
        
        Arguments:
            name {str} -- Name of the measure, used in reports
            dimensions {int} -- Number of dimensions of the result
            
        Keyword Arguments:
            abbreviation {Optional[str]} -- Abbreviation, if empty, then name is used.
            Can be used to define a shorter text representation. (default: {None})
            minimal {Optional[Tuple[float]]} -- Minimal value of the measure. If None, then the measure is not bounded from below. (default: {None})
            maximal {Optional[Tuple[float]]} -- Maximal value of the measure. If None, then the measure is not bounded from above. (default: {None})
            labels {Optional[Tuple[str]]} -- Labels for each dimension. (default: {None})
            trait {Optional[str]} -- Trait of the data, used for specification . Defaults to None.
            
        """

        assert(dimensions > 1)
        super().__init__(name, abbreviation, trait)
        self._dimensions = dimensions
        self._minimal = minimal
        self._maximal = maximal
        self._labels = labels

    @property
    def dimensions(self):
        """Number of dimensions of the result"""
        return self._dimensions

    def minimal(self, i):
        """Minimal value of the i-th dimension. If None, then the measure is not bounded from below."""
        return self._minimal[i]

    def maximal(self, i):
        """Maximal value of the i-th dimension. If None, then the measure is not bounded from above."""
        return self._maximal[i]

    def label(self, i):
        """Label for the i-th dimension."""
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
        """Constructor for Plot class.
        
        Arguments:
            name {str} -- Name of the measure, used in reports
            
        Keyword Arguments:
            abbreviation {Optional[str]} -- Abbreviation, if empty, then name is used.
            Can be used to define a shorter text representation. (default: {None})
            wrt {str} -- Unit of the independent variable. (default: {"frames"})
            minimal {Optional[float]} -- Minimal value of the measure. If None, then the measure is not bounded from below. (default: {None})
            maximal {Optional[float]} -- Maximal value of the measure. If None, then the measure is not bounded from above. (default: {None})
            trait {Optional[str]} -- Trait of the data, used for specification . Defaults to None.
            
        """
        super().__init__(name, abbreviation, trait)
        self._wrt = wrt
        self._minimal = minimal
        self._maximal = maximal

    @property
    def minimal(self):
        """Minimal value of the measure. If None, then the measure is not bounded from below."""
        return self._minimal

    @property
    def maximal(self):
        """Maximal value of the measure. If None, then the measure is not bounded from above."""
        return self._maximal

    @property
    def wrt(self):
        """Unit of the independent variable."""
        return self._wrt

class Curve(Multidimensional):
    """Curve is a list of 2+ dimensional results. The number of elements in a list can vary between samples.
    """

class Analysis(Attributee):
    """Base class for all analysis classes. Analysis is a class that descibes computation of one or more performance metrics for a given experiment."""

    name = String(default=None, description="Name of the analysis")

    def __init__(self, **kwargs):
        """Constructor for Analysis class.
        
        Keyword Arguments:
            name {str} -- Name of the analysis (default: {None})
        """
        super().__init__(**kwargs)
        self._identifier_cache = None

    def compatible(self, experiment: Experiment):
        """Checks if the analysis is compatible with the experiment type."""
        raise NotImplementedError()

    @property
    def title(self) -> str:
        """Returns the title of the analysis. If name is not set, then the default title is returned."""

        if self.name is None:
            return self._title_default
        else:
            return self.name

    @property
    def _title_default(self) -> str:
        """Returns the default title of the analysis. This is used when name is not set."""
        raise NotImplementedError()

    def dependencies(self) -> List["Analysis"]:
        """Returns a list of dependencies of the analysis. This is used to determine the order of execution of the analysis."""
        return []

    @property
    def identifier(self) -> str:
        """Returns a unique identifier of the analysis. This is used to determine if the analysis has been already computed."""

        if not self._identifier_cache is None:
            return self._identifier_cache

        params = self.dump()
        del params["name"]

        confighash = arg_hash(**params)

        self._identifier_cache = class_fullname(self) + "@" + confighash
        
        return self._identifier_cache

    def describe(self) -> Tuple["Result"]:
        """Returns a tuple of descriptions of results of the analysis."""
        raise NotImplementedError()

    def compute(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence], dependencies: List[Grid]) -> Grid:
        """Computes the analysis for the given experiment, trackers and sequences. The dependencies are the results of the dependnent analyses.
        The result is a grid with the results of the analysis. The grid is indexed by trackers and sequences. The axes are described by the axes() method.
        
        Args:
            experiment (Experiment): Experiment to compute the analysis for.
            trackers (List[Tracker]): List of trackers to compute the analysis for.
            sequences (List[Sequence]): List of sequences to compute the analysis for.
            dependencies (List[Grid]): List of dependencies of the analysis.
            
        Returns: Grid with the results of the analysis.
        """
        raise NotImplementedError()

    @property
    def axes(self) -> Axes:
        """ Returns axes semantic description for the result grid """
        raise NotImplementedError()

    def commit(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence]):
        """Commits the analysis for execution on default processor."""
        return AnalysisProcessor.commit_default(self, experiment, trackers, sequences)

    def run(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence]):
        """Runs the analysis on default processor."""
        return AnalysisProcessor.run_default(self, experiment, trackers, sequences)

class SeparableAnalysis(Analysis):
    """Analysis that is separable with respect to trackers and/or sequences, each part can be processed in parallel
    as a separate job. The separation is determined by the result of the axes() method: Axes.BOTH means separation
    in tracker-sequence pairs, Axes.TRACKER means separation according to trackers and Axes.SEQUENCE means separation
    according to sequences.
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

        Returns:
            Tuple[Any]: Tuple of results of the analysis
        """
        raise NotImplementedError()

    def __init__(self, **kwargs):
        """Initializes the analysis. The axes semantic description is checked to be compatible with the dependencies."""
        super().__init__(**kwargs)

        # All dependencies should be mappable to individual parts. If parts contain 
        # separation only across trackers or sequences then we are unable to properly
        # assign dependencies that contain individual 
        if self.axes != Axes.BOTH:
            assert all([dependency.axes != Axes.BOTH for dependency in self.dependencies()])

    def separate(self, trackers: List[Tracker], sequences: List[Sequence]) -> List["SeparablePart"]:
        """Separates the analysis into parts that can be processed separately.
        
        Args:
            trackers (List[Tracker]): List of trackers to compute the analysis for.
            sequences (List[Sequence]): List of sequences to compute the analysis for.
            
        Returns: List of parts of the analysis.

        """
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
        """Joins the results of the analysis into a single grid. The results are indexed by trackers and sequences.

        Args:
            trackers (List[Tracker]): List of trackers to compute the analysis for.
            sequences (List[Sequence]): List of sequences to compute the analysis for.
            results (List[Tuple[Any]]): List of results of the analysis.

        Returns:
            Grid: Grid with the results of the analysis.
        """

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
        """Returns the axes of the analysis. This is used to determine how the analysis is split into parts."""
        return Axes.BOTH

class SequenceAggregator(Analysis): # pylint: disable=W0223
    """Base class for sequence aggregators. Sequence aggregators take the results of a tracker and aggregate them over sequences."""

    def __init__(self, **kwargs):
        """Base constructor."""
        super().__init__(**kwargs)
        # We only support one dependency in aggregator ...
        assert len(self.dependencies()) == 1
        # ... it should produce a grid of results that can be averaged over sequences
        assert self.dependencies()[0].axes == Axes.BOTH

    @abstractmethod
    def aggregate(self, tracker: Tracker, sequences: List[Sequence], results: Grid) -> Tuple[Any]:
        """Aggregate the results of the analysis over sequences for a single tracker.
        
        Args:
            tracker (Tracker): Tracker to aggregate the results for.
            sequences (List[Sequence]): List of sequences to aggregate the results for.
            results (Grid): Results of the analysis for the tracker and sequences.
            
        """
        raise NotImplementedError()

    def compute(self, _: Experiment, trackers: List[Tracker], sequences: List[Sequence], dependencies: List[Grid]) -> Grid:
        """Compute the analysis for a list of trackers and sequences.
        
        Args:
            trackers (List[Tracker]): List of trackers to compute the analysis for.
            sequences (List[Sequence]): List of sequences to compute the analysis for.
            dependencies (List[Grid]): List of dependencies, should be one grid with results of the dependency analysis.
            
        Returns:
            Grid: Grid with the results of the analysis.
        """
        results = dependencies[0]
        transformed_results = Grid(len(trackers), 1)

        for i, tracker in enumerate(trackers):
            transformed_results[i, 0] = self.aggregate(tracker, sequences, results.row(i))

        return transformed_results

    @property
    def axes(self) -> Axes:
        """The analysis is separable in trackers."""
        return Axes.TRACKERS

class TrackerSeparableAnalysis(SeparableAnalysis):
    """Separate analysis into multiple per-tracker tasks, each of them is non-separable.
    """

    @abstractmethod
    def subcompute(self, experiment: Experiment, tracker: Tracker, sequences: List[Sequence], dependencies: List[Grid]) -> Tuple[Any]:
        """Compute the analysis for a single tracker."""
        raise NotImplementedError()

    @property
    def axes(self) -> Axes:
        """The analysis is separable in trackers."""
        return Axes.TRACKERS

class SequenceSeparableAnalysis(SeparableAnalysis):
    """Separate analysis into multiple per-tracker tasks, each of them is non-separable.
    """

    @abstractmethod
    def subcompute(self, experiment: Experiment, trackers: List[Tracker], sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:
        """Compute the analysis for a single sequence."""
        raise NotImplementedError

    @property
    def axes(self) -> Axes:
        """The analysis is separable in sequences."""
        return Axes.SEQUENCES

def is_special(region: Region, code=None) -> bool:
    """Check if the region is special (not a shape) and optionally if it has a specific code."""
    if code is None:
        return region.type == RegionType.SPECIAL
    return region.type == RegionType.SPECIAL and region.code == code

from .processor import process_stack_analyses, AnalysisProcessor, AnalysisError
for module in [".multistart", ".supervised", ".accuracy", ".failures", ".longterm"]:
    importlib.import_module(module, package="vot.analysis")
