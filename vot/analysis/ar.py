import math
from typing import List, Tuple

import numpy as np

from vot.tracker import Tracker, Trajectory
from vot.dataset import Sequence
from vot.dataset.proxy import FrameMapSequence
from vot.experiment import Experiment
from vot.experiment.multirun import SupervisedExperiment
from vot.experiment.multistart import MultiStartExperiment, find_anchors
from vot.region import Region, Special, calculate_overlaps
from vot.analysis import SequenceAveragingAnalysis, \
    MissingResultsException, Measure, Point, is_special, Sorting, simplejoin, Axis
from vot.utilities import alias
from vot.utilities.attributes import Integer, Boolean, Float

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

def count_failures(trajectory: List[Region]) -> Tuple[int, int]:
    return len([region for region in trajectory if is_special(region, Special.FAILURE)]), len(trajectory)

@alias("AccuracyRobustness", "ar", "AccuracyRobustnessSupervised")
class AccuracyRobustness(SequenceAveragingAnalysis):

    sensitivity = Float(default=30, val_min=1)
    burnin = Integer(default=10, val_min=0)
    ignore_unknown = Boolean(default=True)
    bounded = Boolean(default=True)

    @property
    def title(self):
        return "AR analysis"

    def describe(self):
        return Measure("Accuracy", "A", minimal=0, maximal=1, direction=Sorting.DESCENDING), \
             Measure("Robustness", "R", minimal=0, direction=Sorting.ASCENDING), \
             Point("AR plot", dimensions=2, abbreviation="AR", minimal=(0, 0), \
                maximal=(1, 1), labels=("Robustness", "Accuracy"), trait="ar"), \
             None

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, SupervisedExperiment)

    def collapse(self, _: Tracker, sequences: List[Sequence], results: List[tuple]):
        failures = 0
        accuracy = 0
        weight_total = 0

        for a, f, _, w in results:
            failures += f * w
            accuracy += a * w
            weight_total += w

        ar = (math.exp(- (failures / weight_total) * self.sensitivity), accuracy / weight_total)

        return accuracy / weight_total, failures / weight_total, ar, weight_total

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence):
        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        accuracy = 0
        failures = 0
        for trajectory in trajectories:
            failures += count_failures(trajectory.regions())[0]
            accuracy += compute_accuracy(trajectory.regions(), sequence, self.burnin, self.ignore_unknown, self.bounded)[0]

        ar = (math.exp(- (float(failures) / len(trajectories)) * self.sensitivity), accuracy / len(trajectories))

        return accuracy / len(trajectories), failures / len(trajectories), ar, len(trajectories[0])

@alias("AccuracyRobustnessMultiStart", "ar_multistart")
class AccuracyRobustnessMultiStart(SequenceAveragingAnalysis):

    burnin = Integer(default=10, val_min=0)
    grace = Integer(default=10, val_min=0)
    bounded = Boolean(default=True)
    threshold = Float(default=0.1, val_min=0, val_max=1)

    @property
    def title(self):
        return "AR analysis"

    def describe(self):
        return Measure("Accuracy", "A", minimal=0, maximal=1, direction=Sorting.DESCENDING), \
             Measure("Robustness", "R", minimal=0, direction=Sorting.DESCENDING), \
             Point("AR plot", dimensions=2, abbreviation="AR",
                minimal=(0, 0), maximal=(1, 1), labels=("Robustness", "Accuracy"), trait="ar"), \
             None, None

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiStartExperiment)

    def collapse(self, tracker: Tracker, sequences: List[Sequence], results: List[tuple]):
        total_accuracy = 0
        total_robustness = 0
        weight_accuracy = 0
        weight_robustness = 0

        for accuracy, robustness, _, accuracy_w, robustness_w in results:
            total_accuracy += accuracy * accuracy_w
            total_robustness += robustness * robustness_w
            weight_accuracy += accuracy_w
            weight_robustness += robustness_w

        ar = (total_robustness / weight_robustness, total_accuracy / weight_accuracy)

        return total_accuracy / weight_accuracy, total_robustness / weight_robustness, ar, weight_accuracy, weight_robustness

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence):

        results = experiment.results(tracker, sequence)

        forward, backward = find_anchors(sequence, experiment.anchor)

        if not forward and not backward:
            raise RuntimeError("Sequence does not contain any anchors")

        robustness = 0
        accuracy = 0
        total = 0
        for i, reverse in [(f, False) for f in forward] + [(f, True) for f in backward]:
            name = "%s_%08d" % (sequence.name, i)

            if not Trajectory.exists(results, name):
                raise MissingResultsException()

            if reverse:
                proxy = FrameMapSequence(sequence, list(reversed(range(0, i + 1))))
            else:
                proxy = FrameMapSequence(sequence, list(range(i, sequence.length)))

            trajectory = Trajectory.read(results, name)

            overlaps = calculate_overlaps(trajectory.regions(), proxy.groundtruth(), (proxy.size) if self.burnin else None)

            grace = self.grace
            progress = len(proxy)

            for j, overlap in enumerate(overlaps):
                if overlap <= self.threshold and not proxy.groundtruth(j).is_empty():
                    grace = grace - 1
                    if grace == 0:
                        progress = j + 1 - self.grace  # subtract since we need actual point of the failure
                        break
                else:
                    grace = self.grace

            robustness += progress  # simplified original equation: len(proxy) * (progress / len(proxy))
            accuracy += sum(overlaps[0:progress])
            total += len(proxy)

        ar = (robustness / total, accuracy / robustness if robustness > 0 else 0)

        return accuracy / robustness if robustness > 0 else 0, robustness / total, ar, robustness, len(sequence)
