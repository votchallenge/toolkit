from typing import List, Tuple, Any

import numpy as np

from attributee import Boolean, Integer, Include, Float

from vot.analysis import (Measure,
                          MissingResultsException,
                          SequenceAggregator, Sorting,
                          is_special, SeparableAnalysis,
                          analysis_registry)
from vot.dataset import Sequence
from vot.experiment import Experiment
from vot.experiment.multirun import (MultiRunExperiment)
from vot.region import Region, Special, calculate_overlaps
from vot.tracker import Tracker, Trajectory
from vot.utilities.data import Grid

def compute_accuracy(trajectory: List[Region], groundtruth: List[Region], burnin: int = 10, 
    ignore_unknown: bool = True, bounds = None, threshold: float = None) -> float:

    overlaps = np.array(calculate_overlaps(trajectory, groundtruth, bounds))
    mask = np.ones(len(overlaps), dtype=bool)

    if threshold is None: threshold = -1

    for i, (region_tr, region_gt) in enumerate(zip(trajectory, groundtruth)):
        # Skip if groundtruth is unknown or target is not visible
        if region_gt.is_empty() and ignore_unknown:
            mask[i] = False
        if is_special(region_tr, Trajectory.UNKNOWN) and ignore_unknown:
            mask[i] = False
        elif is_special(region_tr, Trajectory.INITIALIZATION):
            for j in range(i, min(len(trajectory), i + burnin)):
                mask[j] = False
        elif is_special(region_tr, Trajectory.FAILURE):
            mask[i] = False
        elif overlaps[i] <= threshold:
            mask[i] = False

    if any(mask):
        return np.mean(overlaps[mask]), np.sum(mask)
    else:
        return 0, 0


@analysis_registry.register("accuracy")
class SequenceAccuracy(SeparableAnalysis):

    burnin = Integer(default=10, val_min=0)
    ignore_unknown = Boolean(default=True)
    bounded = Boolean(default=True)
    threshold = Float(default=None, val_min=0, val_max=1)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "Sequence accurarcy"

    def describe(self):
        return Measure("Accuracy", "AUC", 0, 1, Sorting.DESCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:

        assert isinstance(experiment, MultiRunExperiment)

        objects = sequence.objects()
        objects_accuracy = 0
        bounds = (sequence.size) if self.bounded else None

        for object in objects:
            trajectories = experiment.gather(tracker, sequence, objects=[object])
            if len(trajectories) == 0:
                raise MissingResultsException()

            cummulative = 0

            for trajectory in trajectories:
                accuracy, _ = compute_accuracy(trajectory.regions(), sequence.object(object), self.burnin, self.ignore_unknown, bounds=bounds)
                cummulative += accuracy
            objects_accuracy += cummulative / len(trajectories)

        return objects_accuracy / len(objects),

@analysis_registry.register("average_accuracy")
class AverageAccuracy(SequenceAggregator):

    analysis = Include(SequenceAccuracy)
    weighted = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "Average accurarcy"

    def dependencies(self):
        return self.analysis,

    def describe(self):
        return Measure("Accuracy", "AUC", 0, 1, Sorting.DESCENDING),

    def aggregate(self, _: Tracker, sequences: List[Sequence], results: Grid):
        accuracy = 0
        frames = 0

        for i, sequence in enumerate(sequences):
            if results[i, 0] is None:
                continue

            if self.weighted:
                accuracy += results[i, 0][0] * len(sequence)
                frames += len(sequence)
            else:
                accuracy += results[i, 0][0]
                frames += 1

        return accuracy / frames,
