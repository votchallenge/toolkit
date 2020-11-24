""" This module constans common analysis routines for supervised experiment, e.g. Accuracy-Robustness and EAO as
 defined in VOT papers.
"""

import math
from typing import List, Tuple, Any

import numpy as np

from attributee import Integer, Boolean, Float, Include

from vot.tracker import Tracker, Trajectory
from vot.dataset import Sequence
from vot.dataset.proxy import FrameMapSequence
from vot.experiment import Experiment
from vot.experiment.multirun import SupervisedExperiment
from vot.experiment.multistart import MultiStartExperiment, find_anchors
from vot.region import Region, Special, calculate_overlaps
from vot.analysis import MissingResultsException, Measure, Point, is_special, Plot, Analysis, \
    Sorting, SeparableAnalysis, SequenceAggregator, analysis_registry, TrackerSeparableAnalysis, Axes
from vot.utilities.data import Grid

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


def locate_failures_inits(trajectory: List[Region]) -> Tuple[int, int]:
    return [i for i, region in enumerate(trajectory) if is_special(region, Special.FAILURE)], \
            [i for i, region in enumerate(trajectory) if is_special(region, Special.INITIALIZATION)]

def compute_eao_curve(overlaps: List, weights: List[float], success: List[bool]):
    max_length = max([len(el) for el in overlaps])
    total_runs = len(overlaps)
    
    overlaps_array = np.zeros((total_runs, max_length), dtype=np.float32)
    mask_array = np.zeros((total_runs, max_length), dtype=np.float32)  # mask out frames which are not considered in EAO calculation
    weights_vector = np.reshape(np.array(weights, dtype=np.float32), (len(weights), 1))  # weight of each run

    for i, (o, success) in enumerate(zip(overlaps, success)):
        overlaps_array[i, :len(o)] = np.array(o)
        if not success:
            # tracker has failed during this run - fill zeros until the end of the run
            mask_array[i, :] = 1
        else:
            # tracker has successfully tracked to the end - consider only this part of the sequence
            mask_array[i, :len(o)] = 1

    overlaps_array_sum = overlaps_array.copy()
    for j in range(1, overlaps_array_sum.shape[1]):
        overlaps_array_sum[:, j] = np.mean(overlaps_array[:, 1:j+1], axis=1)
    
    return np.sum(weights_vector * overlaps_array_sum * mask_array, axis=0) / np.sum(mask_array * weights_vector, axis=0).tolist()
    

@analysis_registry.register("supervised_ar")
class AccuracyRobustness(SeparableAnalysis):

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

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:
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

@analysis_registry.register("supervised_average_ar")
class AverageAccuracyRobustness(SequenceAggregator):

    analysis = Include(AccuracyRobustness)

    @property
    def title(self):
        return "AR Analysis"

    def dependencies(self):
        return self.analysis,

    def describe(self):
        return Measure("Accuracy", "A", minimal=0, maximal=1, direction=Sorting.DESCENDING), \
             Measure("Robustness", "R", minimal=0, direction=Sorting.ASCENDING), \
             Point("AR plot", dimensions=2, abbreviation="AR", minimal=(0, 0), \
                maximal=(1, 1), labels=("Robustness", "Accuracy"), trait="ar"), \
             None

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, SupervisedExperiment)

    def aggregate(self, tracker: Tracker, sequences: List[Sequence], results: Grid):
        failures = 0
        accuracy = 0
        weight_total = 0

        for a, f, _, w in results:
            failures += f * w
            accuracy += a * w
            weight_total += w

        ar = (math.exp(- (failures / weight_total) * self.analysis.sensitivity), accuracy / weight_total)

        return accuracy / weight_total, failures / weight_total, ar, weight_total

@analysis_registry.register("supervised_eao_curve")
class EAOCurve(TrackerSeparableAnalysis):

    burnin = Integer(default=10, val_min=0)
    bounded = Boolean(default=True)

    @property
    def title(self):
        return "EAO Curve"

    def describe(self):
        return Plot("Expected Average Overlap", "EAO", minimal=0, maximal=1, trait="eao"),

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, SupervisedExperiment)

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequences: List[Sequence], dependencies: List[Grid]) -> Tuple[Any]:

        overlaps_all = []
        weights_all = []
        success_all = []

        for sequence in sequences:

            trajectories = experiment.gather(tracker, sequence)

            if len(trajectories) == 0:
                raise MissingResultsException()

            for trajectory in trajectories:

                overlaps = calculate_overlaps(trajectory.regions(), sequence.groundtruth(), (sequence.size) if self.bounded else None)
                fail_idxs, init_idxs = locate_failures_inits(trajectory.regions())

                if len(fail_idxs) > 0:

                    for i in range(len(fail_idxs)):
                        overlaps_all.append(overlaps[init_idxs[i]:fail_idxs[i]])
                        success_all.append(False)
                        weights_all.append(1)

                    # handle last initialization
                    if len(init_idxs) > len(fail_idxs):
                        # tracker was initilized, but it has not failed until the end of the sequence
                        overlaps_all.append(overlaps[init_idxs[-1]:])
                        success_all.append(True)
                        weights_all.append(1)

                else:
                    overlaps_all.append(overlaps)
                    success_all.append(True)
                    weights_all.append(1)

        return compute_eao_curve(overlaps_all, weights_all, success_all),

@analysis_registry.register("supervised_eao_score")
class EAOScore(Analysis):

    eaocurve = Include(EAOCurve)
    low = Integer()
    high = Integer()

    @property
    def title(self):
        return "EAO analysis"

    def describe(self):
        return Measure("Expected average overlap", "EAO", 0, 1, Sorting.DESCENDING),

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, SupervisedExperiment)

    def dependencies(self):
        return self.eaocurve,

    def compute(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence], dependencies: List[Grid]) -> Grid:
        return dependencies[0].foreach(lambda x, i, j: (float(np.mean(x[0][self.low:self.high + 1])), ) )

    @property
    def axes(self):
        return Axes.TRACKERS