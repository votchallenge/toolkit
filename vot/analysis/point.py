"""This module contains the implementation of point tracking performance measures."""

import numpy as np
from typing import List, Tuple, Any

from attributee import Include

from vot.tracker import Tracker, Trajectory
from vot.dataset import Sequence
from vot.experiment import Experiment
from vot.experiment.multirun import UnsupervisedExperiment
from vot.region import RegionType
from vot.analysis import SeparableAnalysis, SequenceAggregator, \
    MissingResultsException, Measure, Sorting, is_special
from vot.utilities.data import Grid

THRESHOLDS = [1, 2, 4, 8, 16]

def compute_point_accuracy(predicted: list, groundtruth: list, width: int, height: int) -> Tuple[float, ...]:
    """Compute per-threshold accuracy and d_avg for a single point trajectory.

    For each frame where both prediction and groundtruth are valid points,
    the L2 distance is computed in a normalized coordinate space equivalent
    to 256x256 pixels, making the thresholds resolution-independent.

    Args:
        predicted (list): Predicted trajectory as a list of regions.
        groundtruth (list): Groundtruth trajectory as a list of regions.
        width (int): Frame width in pixels.
        height (int): Frame height in pixels.

    Returns:
        Tuple of (d_avg, d_1, d_2, d_4, d_8, d_16, n_evaluated) where each d_t
        is the fraction of frames where normalized L2 error < t, and n_evaluated
        is the number of frames included in the computation.
    """
    sx = (width - 1) / 255.0
    sy = (height - 1) / 255.0

    counts = np.zeros(len(THRESHOLDS), dtype=np.float64)
    n = 0

    for pred, gt in zip(predicted, groundtruth):
        if pred.type != RegionType.POINT or gt.type != RegionType.POINT:
            continue
        dx = (pred.x - gt.x) / sx
        dy = (pred.y - gt.y) / sy
        dist = np.sqrt(dx * dx + dy * dy)
        for i, thr in enumerate(THRESHOLDS):
            if dist < thr:
                counts[i] += 1
        n += 1

    if n == 0:
        return (0.0,) * (len(THRESHOLDS) + 1) + (0,)

    fractions = counts / n
    d_avg = float(np.mean(fractions))
    return (d_avg,) + tuple(float(f) for f in fractions) + (n,)


class PointAccuracy(SeparableAnalysis):
    """Per-sequence point tracking accuracy. Computes d_avg and per-threshold
    accuracy scores for each tracker on each sequence."""

    @property
    def _title_default(self):
        """Returns the default title of the analysis."""
        return "Point accuracy"

    def describe(self):
        """Returns descriptions of the results produced by this analysis."""
        return (
            Measure("Average point accuracy", "d_avg", minimal=0, maximal=1, direction=Sorting.DESCENDING),
            Measure("Point accuracy at 1px", "d_1", minimal=0, maximal=1, direction=Sorting.DESCENDING),
            Measure("Point accuracy at 2px", "d_2", minimal=0, maximal=1, direction=Sorting.DESCENDING),
            Measure("Point accuracy at 4px", "d_4", minimal=0, maximal=1, direction=Sorting.DESCENDING),
            Measure("Point accuracy at 8px", "d_8", minimal=0, maximal=1, direction=Sorting.DESCENDING),
            Measure("Point accuracy at 16px", "d_16", minimal=0, maximal=1, direction=Sorting.DESCENDING),
            None,  # frame count, used for weighted aggregation
        )

    def compatible(self, experiment: Experiment):
        """Only compatible with unsupervised experiments."""
        return isinstance(experiment, UnsupervisedExperiment)

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:
        """Compute point accuracy for a single tracker on a single sequence.

        Args:
            experiment (Experiment): The experiment.
            tracker (Tracker): The tracker.
            sequence (Sequence): The sequence.
            dependencies (List[Grid]): Unused.

        Returns:
            Tuple of (d_avg, d_1, d_2, d_4, d_8, d_16, n_frames).
        """
        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        width, height = sequence.size

        d_avg_sum = np.zeros(len(THRESHOLDS) + 1, dtype=np.float64)
        n_total = 0

        for o in sequence.objects():
            gt = [sequence.frame(i).object(o) for i in range(len(sequence))]
            trajectories_o = experiment.gather(tracker, sequence, objects=[o])
            if len(trajectories_o) == 0:
                continue
            pred = trajectories_o[0].regions()
            result = compute_point_accuracy(pred, gt, width, height)
            n = result[-1]
            if n > 0:
                for k in range(len(THRESHOLDS) + 1):
                    d_avg_sum[k] += result[k] * n
                n_total += n

        if n_total == 0:
            return (0.0,) * (len(THRESHOLDS) + 1) + (0,)

        averaged = tuple(float(d_avg_sum[k] / n_total) for k in range(len(THRESHOLDS) + 1))
        return averaged + (n_total,)


class AveragePointAccuracy(SequenceAggregator):
    """Dataset-level point tracking accuracy. Aggregates per-sequence results
    from PointAccuracy into a single weighted average over all sequences."""

    analysis = Include(PointAccuracy)

    @property
    def _title_default(self):
        """Returns the default title of the analysis."""
        return "Average point accuracy"

    def dependencies(self):
        """Returns the dependency on PointAccuracy."""
        return self.analysis,

    def describe(self):
        """Returns descriptions of the results produced by this analysis."""
        return (
            Measure("Average point accuracy", "d_avg", minimal=0, maximal=1, direction=Sorting.DESCENDING),
            Measure("Point accuracy at 1px", "d_1", minimal=0, maximal=1, direction=Sorting.DESCENDING),
            Measure("Point accuracy at 2px", "d_2", minimal=0, maximal=1, direction=Sorting.DESCENDING),
            Measure("Point accuracy at 4px", "d_4", minimal=0, maximal=1, direction=Sorting.DESCENDING),
            Measure("Point accuracy at 8px", "d_8", minimal=0, maximal=1, direction=Sorting.DESCENDING),
            Measure("Point accuracy at 16px", "d_16", minimal=0, maximal=1, direction=Sorting.DESCENDING),
            None,
        )

    def compatible(self, experiment: Experiment):
        """Only compatible with unsupervised experiments."""
        return isinstance(experiment, UnsupervisedExperiment)

    def aggregate(self, tracker: Tracker, sequences: List[Sequence], results: Grid) -> Tuple[Any]:
        """Aggregate per-sequence results into dataset-level scores.

        Args:
            tracker (Tracker): The tracker.
            sequences (List[Sequence]): All sequences.
            results (Grid): Per-sequence results from PointAccuracy.

        Returns:
            Tuple of (d_avg, d_1, d_2, d_4, d_8, d_16, total_frames).
        """
        n_measures = len(THRESHOLDS) + 1
        weighted_sum = np.zeros(n_measures, dtype=np.float64)
        n_total = 0

        for result in results:
            if result is None:
                continue
            n = result[-1]
            if n > 0:
                for k in range(n_measures):
                    weighted_sum[k] += result[k] * n
                n_total += n

        if n_total == 0:
            return (0.0,) * n_measures + (0,)

        averaged = tuple(float(weighted_sum[k] / n_total) for k in range(n_measures))
        return averaged + (n_total,)
