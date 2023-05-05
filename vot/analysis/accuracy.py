from typing import List, Tuple, Any

import numpy as np

from attributee import Boolean, Integer, Include, Float

from vot.analysis import (Measure,
                          MissingResultsException,
                          SequenceAggregator, Sorting,
                          is_special, SeparableAnalysis,
                          analysis_registry, Curve)
from vot.dataset import Sequence
from vot.experiment import Experiment
from vot.experiment.multirun import (MultiRunExperiment)
from vot.region import Region, calculate_overlaps
from vot.tracker import Tracker, Trajectory
from vot.utilities.data import Grid

def gather_overlaps(trajectory: List[Region], groundtruth: List[Region], burnin: int = 10, 
    ignore_unknown: bool = True, ignore_invisible: bool = False, bounds = None, threshold: float = None) -> float:

    overlaps = np.array(calculate_overlaps(trajectory, groundtruth, bounds))
    mask = np.ones(len(overlaps), dtype=bool)

    if threshold is None: threshold = -1

    for i, (region_tr, region_gt) in enumerate(zip(trajectory, groundtruth)):
        # Skip if groundtruth is unknown
        if is_special(region_gt, Sequence.UNKNOWN):
            mask[i] = False
        elif ignore_invisible and region_gt.is_empty():
            mask[i] = False
        # Skip if predicted is unknown
        elif is_special(region_tr, Trajectory.UNKNOWN) and ignore_unknown:
            mask[i] = False
        # Skip if predicted is initialization frame
        elif is_special(region_tr, Trajectory.INITIALIZATION):
            for j in range(i, min(len(trajectory), i + burnin)):
                mask[j] = False
        elif is_special(region_tr, Trajectory.FAILURE):
            mask[i] = False
        elif overlaps[i] <= threshold:
            mask[i] = False

    return overlaps[mask]

@analysis_registry.register("accuracy")
class SequenceAccuracy(SeparableAnalysis):

    burnin = Integer(default=10, val_min=0)
    ignore_unknown = Boolean(default=True)
    ignore_invisible = Boolean(default=False)    
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
                overlaps = gather_overlaps(trajectory.regions(), sequence.object(object), self.burnin, 
                                        ignore_unknown=self.ignore_unknown, ignore_invisible=self.ignore_invisible, bounds=bounds, threshold=self.threshold)
                if overlaps.size > 0:
                    cummulative += np.mean(overlaps)

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
        return "Accurarcy"

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

@analysis_registry.register("success_plot")
class SuccessPlot(SeparableAnalysis):

    ignore_unknown = Boolean(default=True)
    ignore_invisible = Boolean(default=False)
    burnin = Integer(default=0, val_min=0)
    bounded = Boolean(default=True)
    threshold = Float(default=None, val_min=0, val_max=1)
    resolution = Integer(default=100, val_min=2)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "Sequence success plot"

    def describe(self):
        return Curve("Success plot", 2, "Success", minimal=(0, 0), maximal=(1, 1), labels=("Threshold", "Success"), trait="success"),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:

        assert isinstance(experiment, MultiRunExperiment)

        objects = sequence.objects()
        bounds = (sequence.size) if self.bounded else None

        axis_x = np.linspace(0, 1, self.resolution)
        axis_y = np.zeros_like(axis_x)

        for object in objects:
            trajectories = experiment.gather(tracker, sequence, objects=[object])
            if len(trajectories) == 0:
                raise MissingResultsException()

            object_y = np.zeros_like(axis_x) 

            for trajectory in trajectories:
                overlaps = gather_overlaps(trajectory.regions(), sequence.object(object), burnin=self.burnin, ignore_unknown=self.ignore_unknown, ignore_invisible=self.ignore_invisible, bounds=bounds, threshold=self.threshold)

                for i, threshold in enumerate(axis_x):
                    if threshold == 1:
                        # Nicer handling of the edge case
                        object_y[i] += np.sum(overlaps >= threshold) / len(overlaps)
                    else:
                        object_y[i] += np.sum(overlaps > threshold) / len(overlaps)

            axis_y += object_y / len(trajectories)

        axis_y /= len(objects)

        return [(x, y) for x, y in zip(axis_x, axis_y)],


@analysis_registry.register("average_success_plot")
class AverageSuccessPlot(SequenceAggregator):

    resolution = Integer(default=100, val_min=2)
    analysis = Include(SuccessPlot)

    def dependencies(self):
        return self.analysis,

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "Sequence success plot"

    def describe(self):
        return Curve("Success plot", 2, "Success", minimal=(0, 0), maximal=(1, 1), labels=("Threshold", "Success"), trait="success"),

    def aggregate(self, _: Tracker, sequences: List[Sequence], results: Grid):
        axis_x = np.linspace(0, 1, self.resolution)
        axis_y = np.zeros_like(axis_x)

        for i, _ in enumerate(sequences):
            if results[i, 0] is None:
                continue

            curve = results[i, 0][0]

            for j, (_, y) in enumerate(curve):
                axis_y[j] += y

        axis_y /= len(sequences)

        return [(x, y) for x, y in zip(axis_x, axis_y)],
