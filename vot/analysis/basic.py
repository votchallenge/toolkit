from typing import List, Tuple

import numpy as np

from vot.analysis import (DependentAnalysis, Measure,
                          MissingResultsException, Point,
                          SequenceAggregator, Sorting,
                          is_special, FullySeparableAnalysis)
from vot.dataset import Sequence
from vot.experiment import Experiment
from vot.experiment.multirun import (MultiRunExperiment, SupervisedExperiment)
from vot.region import Region, Special, calculate_overlaps
from vot.tracker import Tracker
from vot.utilities import alias
from vot.utilities.data import Grid
from vot.utilities.attributes import Boolean, Integer, Include

def compute_accuracy(trajectory: List[Region], sequence: Sequence, burnin: int = 10, 
    ignore_unknown: bool = True, bounded: bool = True) -> float:

    overlaps = np.array(calculate_overlaps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    mask = np.ones(len(overlaps), dtype=bool)

    for i, region in enumerate(trajectory):
        if is_special(region, Special.UNKNOWN) and ignore_unknown:
            mask[i] = False
        elif is_special(region, Special.INITIALIZATION):
            for j in range(i, min(len(trajectory), i + burnin)):
                mask[j] = False
        elif is_special(region, Special.FAILURE):
            mask[i] = False
    
    if any(mask):
        return np.mean(overlaps[mask]), np.sum(mask)
    else:
        return 0, 0

def compute_eao_partial(overlaps: List, success: List[bool], curve_length: int):
    phi = curve_length * [float(0)]
    active = curve_length * [float(0)]

    for i, (o, success) in enumerate(zip(overlaps, success)):

        o_array = np.array(o)

        for j in range(1, curve_length):

            if j < len(o):
                phi[j] += np.mean(o_array[1:j+1])
                active[j] += 1
            elif not success:
                phi[j] += np.sum(o_array[1:len(o)]) / (j - 1)
                active[j] += 1

    phi = [p / a if a > 0 else 0 for p, a in zip(phi, active)]
    return phi, active

def count_failures(trajectory: List[Region]) -> Tuple[int, int]:
    return len([region for region in trajectory if is_special(region, Special.FAILURE)]), len(trajectory)

class AverageAccuracyPerSequence(FullySeparableAnalysis):

    burnin = Integer(default=10, val_min=0)
    ignore_unknown = Boolean(default=True)
    bounded = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "Average accurarcy"

    def describe(self):
        return Measure("Accuracy", "AUC", 0, 1, Sorting.DESCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence):

        if isinstance(experiment, MultiRunExperiment):
            trajectories = experiment.gather(tracker, sequence)

            if len(trajectories) == 0:
                raise MissingResultsException()

            cummulative = 0
            for trajectory in trajectories:
                accuracy, _ = compute_accuracy(trajectory.regions(), sequence, self.burnin, self.ignore_unknown, self.bounded)
                cummulative = cummulative + accuracy

            return cummulative / len(trajectories),

@alias("auc", "Average Overlap", "AverageAccuracy")
class AverageAccuracy(SequenceAggregator):

    analysis = Include(AverageAccuracyPerSequence)
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
            if results[i] is None:
                continue

            if self.weighted:
                accuracy += results[i][0] * len(sequence)
                frames += len(sequence)
            else:
                accuracy += results[i][0]
                frames += 1

        return accuracy / frames,

class FailuresPerSequence(FullySeparableAnalysis):

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, SupervisedExperiment)

    @property
    def title(self):
        return "Number of failures"

    def describe(self):
        return Measure("Failures", "F", 0, None, Sorting.ASCENDING), 

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence):
        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        failures = 0
        for trajectory in trajectories:
            failures = failures + count_failures(trajectory.regions())[0]

        return failures / len(trajectories), len(trajectories[0])


@alias("failures", "FailureCount", "Failures")
class FailureCount(SequenceAggregator):

    analysis = Include(FailuresPerSequence)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, SupervisedExperiment)

    def dependencies(self):
        return self.analysis, 

    @property
    def title(self):
        return "Number of failures"

    def describe(self):
        return Measure("Failures", "F", 0, None, Sorting.ASCENDING), 

    def aggregate(self, _: Tracker, sequences: List[Sequence], results: Grid):
        failures = 0

        for a in results:
            failures = failures + a[0]

        return failures,
