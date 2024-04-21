"""This module contains the implementation of the FailureCount analysis. The analysis counts the number of failures in one or more sequences."""

from typing import List, Tuple, Any

from attributee import Include

from vot.analysis import (Measure,
                          MissingResultsException,
                          SequenceAggregator, Sorting,
                          is_special, SeparableAnalysis,
                          analysis_registry)
from vot.dataset import Sequence
from vot.experiment import Experiment
from vot.experiment.multirun import (SupervisedExperiment)
from vot.region import Region
from vot.tracker import Tracker
from vot.utilities.data import Grid


def count_failures(trajectory: List[Region]) -> Tuple[int, int]:
    """Count the number of failures in a trajectory. A failure is defined as a region is annotated as Special.FAILURE by the experiment."""
    return len([region for region in trajectory if is_special(region, SupervisedExperiment.FAILURE)]), len(trajectory)


@analysis_registry.register("failures")
class FailureCount(SeparableAnalysis):
    """Count the number of failures in a sequence. A failure is defined as a region is annotated as Special.FAILURE by the experiment."""

    def compatible(self, experiment: Experiment):
        """Check if the experiment is compatible with the analysis."""
        return isinstance(experiment, SupervisedExperiment)

    @property
    def _title_default(self):
        """Default title for the analysis."""
        return "Number of failures"

    def describe(self):
        """Describe the analysis."""
        return Measure("Failures", "F", 0, None, Sorting.ASCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:
        """Compute the analysis for a single sequence."""

        assert isinstance(experiment, SupervisedExperiment)

        objects = sequence.objects()
        objects_failures = 0

        for object in objects:
            trajectories = experiment.gather(tracker, sequence, objects=[object])
            if len(trajectories) == 0:
                raise MissingResultsException()

            failures = 0
            for trajectory in trajectories:
                failures = failures + count_failures(trajectory.regions())[0]
            objects_failures += failures / len(trajectories)

        return objects_failures / len(objects), len(sequence)

@analysis_registry.register("cumulative_failures")
class CumulativeFailureCount(SequenceAggregator):
    """Count the number of failures over all sequences. A failure is defined as a region is annotated as Special.FAILURE by the experiment."""

    analysis = Include(FailureCount)

    def compatible(self, experiment: Experiment):
        """Check if the experiment is compatible with the analysis."""
        return isinstance(experiment, SupervisedExperiment)

    def dependencies(self):
        """Return the dependencies of the analysis."""
        return self.analysis,

    @property
    def _title_default(self):
        """Default title for the analysis."""
        return "Number of failures"

    def describe(self):
        """Describe the analysis."""
        return Measure("Failures", "F", 0, None, Sorting.ASCENDING), 

    def aggregate(self, _: Tracker, sequences: List[Sequence], results: Grid):
        """Aggregate the analysis for a list of sequences. The aggregation is done by summing the number of failures for each sequence.

        Args:
            sequences (List[Sequence]): The list of sequences to aggregate.
            results (Grid): The results of the analysis for each sequence.

        Returns:
            Tuple[Any]: The aggregated analysis.
        """

        failures = 0

        for a in results:
            failures = failures + a[0]

        return failures,
