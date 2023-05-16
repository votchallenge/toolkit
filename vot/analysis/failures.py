
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
from vot.region import Region, Special, calculate_overlaps
from vot.tracker import Tracker
from vot.utilities.data import Grid


def count_failures(trajectory: List[Region]) -> Tuple[int, int]:
    return len([region for region in trajectory if is_special(region, SupervisedExperiment.FAILURE)]), len(trajectory)


@analysis_registry.register("failures")
class FailureCount(SeparableAnalysis):

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, SupervisedExperiment)

    @property
    def _title_default(self):
        return "Number of failures"

    def describe(self):
        return Measure("Failures", "F", 0, None, Sorting.ASCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:

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

        return objects_failures / len(objects), sequence.length

@analysis_registry.register("cumulative_failures")
class CumulativeFailureCount(SequenceAggregator):

    analysis = Include(FailureCount)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, SupervisedExperiment)

    def dependencies(self):
        return self.analysis,

    @property
    def _title_default(self):
        return "Number of failures"

    def describe(self):
        return Measure("Failures", "F", 0, None, Sorting.ASCENDING), 

    def aggregate(self, _: Tracker, sequences: List[Sequence], results: Grid):
        failures = 0

        for a in results:
            failures = failures + a[0]

        return failures,
