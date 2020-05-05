import logging
from typing import List, Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod
from concurrent.futures import Executor

from cachetools import Cache

from vot import VOTException
from vot.tracker import Tracker
from vot.dataset import Sequence
from vot.experiment import Experiment
from vot.region import Region, RegionType
from vot.utilities import class_fullname, arg_hash

class MissingResultsException(VOTException):
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
    def describe(self) -> Tuple["Result"]:
        """Returns a tuple of descriptions of results
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self, tracker: Tracker, experiment: Experiment, sequences: List[Sequence]):
        raise NotImplementedError

class Result(ABC):
    """Abstract result object base. This is the base class for all result descriptions.
    """

    def __init__(self, name: str, abbreviation: Optional[str] = None):
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

    @property
    def name(self):
        return self._name

    @property
    def abbreviation(self):
        return self._abbreviation

class Label(Result):

    def __init__(self, name: str, abbreviation: Optional[str] = None):
        super().__init__(name, abbreviation)

class Measure(Result):
    """Measure describes a single value numerical output of an analysis. Can have minimum and maximum value as well
    as direction of sorting.
    """

    DESCENDING = "descending"
    ASCENDING = "ascending"

    def __init__(self, name: str, abbreviation: Optional[str] = None, minimal: Optional[float] = None, \
        maximal: Optional[float] = None, direction: Optional[str] = ASCENDING):
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

class Point(Result):
    """Point is a two or more dimensional numerical output that can be visualized in a scatter plot.
    """

    def __init__(self, name: str, dimensions: int, abbreviation: Optional[str] = None, minimal: Optional[Tuple[float]] = None, \
        maximal: Optional[Tuple[float]] = None):
        assert(dimensions > 1)
        super().__init__(name, abbreviation)
        self._dimensions = dimensions
        self._minimal = minimal
        self._maximal = maximal

    @property
    def dimensions(self):
        return self._dimensions

    def minimal(self, i):
        return self._minimal[i]

    def maximal(self, i):
        return self._maximal[i]

class Plot(Result):
    """Plot describes a result in form of a list of values with optional minimum and maximum with respect to some unit. The
    results of the same analysis for different trackers should have the same number of measurements (independent variable).
    """

    def __init__(self, name: str, abbreviation: Optional[str] = None, wrt: str = "frames", minimal: Optional[float] = None, \
        maximal: Optional[float] = None):
        super().__init__(name, abbreviation)
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

class Curve(Result):
    """Curve is a list of 2 or more dimensional results. The number of elements in a list can vary.
    """

    def __init__(self, name: str, dimensions: int, abbreviation: Optional[str] = None, minimal: Optional[Tuple[float]] = None, \
        maximal: Optional[Tuple[float]] = None):
        assert(dimensions > 1)
        super().__init__(name, abbreviation)
        self._dimensions = dimensions
        self._minimal = minimal
        self._maximal = maximal

    @property
    def dimensions(self):
        return self._dimensions

    def minimal(self, i):
        return self._minimal[i]

    def maximal(self, i):
        return self._maximal[i]

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

def process_analyses(workspace: "Workspace", trackers: List[Tracker], executor: Executor, cache: Cache):

    from vot.analysis.backend import AnalysisProcessor
    from vot.utilities import Progress
    from threading import Condition

    processor = AnalysisProcessor(executor, cache)

    logger = logging.getLogger("vot")

    results = dict()
    condition = Condition()

    def insert_result(container: dict, key):
        def insert(x):
            if isinstance(x, Exception):
                logger.exception(x)
            else:
                container[key] = x
            with condition:
                condition.notify()
        return insert

    for experiment in workspace.stack:

        logger.debug("Traversing experiment %s", experiment.identifier)

        results[experiment] = dict()

        for analysis in workspace.stack.analyses(experiment):

            if not analysis.compatible(experiment):
                continue

            logger.debug("Traversing analysis %s", analysis.name)

            analysis_results = dict()

            for tracker in trackers:
                with condition:
                    analysis_results[tracker] = None
                processor.submit(analysis, tracker, experiment, workspace.dataset, insert_result(analysis_results, tracker))

            results[experiment][analysis] = analysis_results

    logger.debug("Waiting for %d analysis tasks to finish", processor.total)

    progress = Progress(desc="Analysis", total=processor.total, unit="tasks")

    try:

        while True:
            with condition:
                progress.update_absolute(processor.total - processor.pending)

                if processor.pending == 0:
                    break

                condition.wait(1)

    except KeyboardInterrupt:
        processor.cancel_all()
        progress.close()
        logger.info("Analysis interrupted by user, aborting.")
        return None

    progress.close()

    return results


