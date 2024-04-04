"""Accuracy analysis. Computes average overlap between predicted and groundtruth regions."""

from typing import List, Tuple, Any

import numpy as np

from attributee import Boolean, Integer, Include, Float, String

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
    ignore_unknown: bool = True, ignore_invisible: bool = False, bounds = None, threshold: float = None, ignore_masks: List[Region] = None) -> np.ndarray:
    """Gather overlaps between trajectory and groundtruth regions. 
    
    Args:
        trajectory (List[Region]): List of regions predicted by the tracker.
        groundtruth (List[Region]): List of groundtruth regions.
        burnin (int, optional): Number of frames to skip at the beginning of the sequence. Defaults to 10.
        ignore_unknown (bool, optional): Ignore unknown regions in the groundtruth. Defaults to True.
        ignore_invisible (bool, optional): Ignore invisible regions in the groundtruth. Defaults to False.
        bounds ([type], optional): Bounds of the sequence. Defaults to None.
        threshold (float, optional): Minimum overlap to consider. Defaults to None.
        ignore_masks (List[Region], optional): List of regions to ignore. Defaults to None.
        
    Returns:
        np.ndarray: List of overlaps."""

    assert len(trajectory) == len(groundtruth), "Trajectory and groundtruth must have the same length."

    if ignore_masks is not None:
        assert len(trajectory) == len(ignore_masks), "Trajectory and ignore mask must have the same length."

    overlaps = np.array(calculate_overlaps(trajectory, groundtruth, bounds, ignore=ignore_masks))
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
    """Sequence accuracy analysis. Computes average overlap between predicted and groundtruth regions."""

    burnin = Integer(default=10, val_min=0, description="Number of frames to skip after the initialization.")
    ignore_unknown = Boolean(default=True, description="Ignore unknown regions in the groundtruth.")
    ignore_invisible = Boolean(default=False, description="Ignore invisible regions in the groundtruth.")    
    bounded = Boolean(default=True, description="Consider only the bounded region of the sequence.")
    threshold = Float(default=None, val_min=0, val_max=1, description="Minimum overlap to consider.")
    ignore_masks = String(default="_ignore", description="Object ID used to get ignore masks.")
    filter_tag = String(default=None, description="Filter tag for the analysis.")

    def compatible(self, experiment: Experiment):
        """Check if the experiment is compatible with the analysis."""
        return isinstance(experiment, MultiRunExperiment)

    @property
    def _title_default(self):
        """Default title of the analysis."""
        return "Sequence accurarcy"

    def describe(self):
        """Describe the analysis."""
        return Measure(self.title, "", 0, 1, Sorting.DESCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:
        """Compute the analysis for a single sequence. 
        
        Args:
            experiment (Experiment): Experiment.
            tracker (Tracker): Tracker.
            sequence (Sequence): Sequence.
            dependencies (List[Grid]): List of dependencies.
            
        Returns:
            Tuple[Any]: Tuple of results.
        """
        assert isinstance(experiment, MultiRunExperiment)

        objects = sequence.objects()
        objects_accuracy = 0
        bounds = (sequence.size) if self.bounded else None

        ignore_masks = sequence.object(self.ignore_masks)

        if self.filter_tag is not None:
            frame_mask = [self.filter_tag in sequence.tags(i) for i in range(len(sequence))]
        else:
            frame_mask = None

        for object in objects:
            trajectories = experiment.gather(tracker, sequence, objects=[object])
            if len(trajectories) == 0:
                raise MissingResultsException()

            cummulative = 0

            for trajectory in trajectories:
                if frame_mask is not None:
                    trajectory = [region for region, m in zip(trajectory, frame_mask) if m]
                    groundtruth = [region for region, m in zip(sequence.object(object), frame_mask) if m]
                    masks = [region for region, m in zip(ignore_masks, frame_mask) if m] 
                else:
                    trajectory = trajectory
                    groundtruth = sequence.object(object)
                    masks = ignore_masks
    
                overlaps = gather_overlaps(trajectory, groundtruth, self.burnin, 
                                        ignore_unknown=self.ignore_unknown, ignore_invisible=self.ignore_invisible, bounds=bounds, threshold=self.threshold, ignore_masks=masks)
                if overlaps.size > 0:
                    cummulative += np.mean(overlaps)


            objects_accuracy += cummulative / len(trajectories)

        return objects_accuracy / len(objects),

@analysis_registry.register("average_accuracy")
class AverageAccuracy(SequenceAggregator):
    """Average accuracy analysis. Computes average overlap between predicted and groundtruth regions."""

    analysis = Include(SequenceAccuracy, description="Sequence accuracy analysis.")
    weighted = Boolean(default=True, description="Weight accuracy by the number of frames.")

    def compatible(self, experiment: Experiment):
        """Check if the experiment is compatible with the analysis. This analysis requires a multirun experiment."""
        return isinstance(experiment, MultiRunExperiment)

    @property
    def _title_default(self):
        """Default title of the analysis."""
        return "Accurarcy"

    def dependencies(self):
        """List of dependencies."""
        return self.analysis,

    def describe(self):
        """Describe the analysis."""
        return Measure(self.title, "", 0, 1, Sorting.DESCENDING),

    def aggregate(self, _: Tracker, sequences: List[Sequence], results: Grid):
        """Aggregate the results of the analysis.
        
        Args:    
            tracker (Tracker): Tracker.
            sequences (List[Sequence]): List of sequences.
            results (Grid): Grid of results.
            
        Returns:
            Tuple[Any]: Tuple of results.
        """

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
    """Success plot analysis. Computes the success plot of the tracker."""

    ignore_unknown = Boolean(default=True, description="Ignore unknown regions in the groundtruth.")
    ignore_invisible = Boolean(default=False, description="Ignore invisible regions in the groundtruth.")
    burnin = Integer(default=0, val_min=0, description="Number of frames to skip after the initialization.")
    bounded = Boolean(default=True, description="Consider only the bounded region of the sequence.")
    threshold = Float(default=None, val_min=0, val_max=1, description="Minimum overlap to consider.")
    resolution = Integer(default=100, val_min=2, description="Number of points in the plot.")
    ignore_masks = String(default="_ignore", description="Object ID used to get ignore masks.")
    filter_tag = String(default=None, description="Filter tag for the analysis.")

    def compatible(self, experiment: Experiment):
        """Check if the experiment is compatible with the analysis. This analysis is only compatible with multi-run experiments."""
        return isinstance(experiment, MultiRunExperiment)

    @property
    def _title_default(self):
        """Default title of the analysis."""
        return "Sequence success plot"

    def describe(self):
        """Describe the analysis."""
        return Curve("Plot", 2, "S", minimal=(0, 0), maximal=(1, 1), labels=("Threshold", "Success"), trait="success"),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:
        """Compute the analysis for a single sequence. 

        Args:
            experiment (Experiment): Experiment.
            tracker (Tracker): Tracker.
            sequence (Sequence): Sequence.
            dependencies (List[Grid]): List of dependencies.

        Returns:
            Tuple[Any]: Tuple of results.
        """

        assert isinstance(experiment, MultiRunExperiment)

        objects = sequence.objects()
        bounds = (sequence.size) if self.bounded else None

        axis_x = np.linspace(0, 1, self.resolution)
        axis_y = np.zeros_like(axis_x)

        ignore_masks = sequence.object(self.ignore_masks)

        for object in objects:
            trajectories = experiment.gather(tracker, sequence, objects=[object])
            if len(trajectories) == 0:
                raise MissingResultsException()

            object_y = np.zeros_like(axis_x) 

            if self.filter_tag is not None:
                frame_mask = [self.filter_tag in sequence.tags(i) for i in range(len(sequence))]
            else:
                frame_mask = None

            for trajectory in trajectories:
                if frame_mask is not None:
                    trajectory = [region for region, m in zip(trajectory, frame_mask) if m]
                    groundtruth = [region for region, m in zip(sequence.object(object), frame_mask) if m] 
                    masks = [region for region, m in zip(ignore_masks, frame_mask) if m]
                else:
                    groundtruth = sequence.object(object)
                    masks = ignore_masks
                
                print(len(trajectory), len(groundtruth), len(masks), len(ignore_masks), len(sequence))
                
                overlaps = gather_overlaps(trajectory, groundtruth, burnin=self.burnin, ignore_unknown=self.ignore_unknown, 
                                            ignore_invisible=self.ignore_invisible, bounds=bounds, threshold=self.threshold, ignore_masks=masks)

                for i, threshold in enumerate(axis_x):
                    if threshold == 1:
                        # Nicer handling of the edge case
                        object_y[i] += np.sum(overlaps >= threshold) / len(overlaps)
                    else:
                        object_y[i] += np.sum(overlaps > threshold) / len(overlaps)
                print(sequence.name, object_y, len(overlaps))

            axis_y += object_y / len(trajectories)

        axis_y /= len(objects)

        return [(x, y) for x, y in zip(axis_x, axis_y)],


@analysis_registry.register("average_success_plot")
class AverageSuccessPlot(SequenceAggregator):
    """Average success plot analysis. Computes the average success plot of the tracker."""

    resolution = Integer(default=100, val_min=2)
    analysis = Include(SuccessPlot)

    def dependencies(self):
        """List of dependencies."""
        return self.analysis,

    def compatible(self, experiment: Experiment):
        """Check if the experiment is compatible with the analysis. This analysis is only compatible with multi-run experiments."""
        return isinstance(experiment, MultiRunExperiment)

    @property
    def _title_default(self):
        """Default title of the analysis."""
        return "Success plot"

    def describe(self):
        """Describe the analysis."""
        return Curve("Plot", 2, "S", minimal=(0, 0), maximal=(1, 1), labels=("Threshold", "Success"), trait="success"),

    def aggregate(self, _: Tracker, sequences: List[Sequence], results: Grid):
        """Aggregate the results of the analysis.
        
        Args:    
            tracker (Tracker): Tracker. 
            sequences (List[Sequence]): List of sequences.
            results (Grid): Grid of results.
            
        Returns:
            Tuple[Any]: Tuple of results.
        """
        
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
