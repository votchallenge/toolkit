"""Video object segmentation performance measures.

This module implements the DAVIS-style region similarity (J), contour accuracy
(F), and their mean (J&F). Inputs are frame-wise region sequences. Regions are
rasterized to the supplied image size for contour evaluation; ``Mask`` regions
are the intended input, but other toolkit shapes are supported as well.
"""

from math import ceil, sqrt
from typing import List, Sequence, Tuple

import numpy as np
from attributee import Boolean

from vot.analysis import Measure, MissingResultsException, SeparableAnalysis, Sorting
from vot.analysis.mot import _gather_multiobject_runs
from vot.dataset import Sequence as DatasetSequence
from vot.experiment import Experiment
from vot.experiment.multirun import MultiRunExperiment
from vot.region import Region, Special, calculate_overlap
from vot.tracker import Tracker
from vot.utilities.data import Grid


def _validate_inputs(groundtruth: Sequence[Region], prediction: Sequence[Region]):
    if len(groundtruth) != len(prediction):
        raise ValueError("Groundtruth and prediction must have the same number of frames")


def _mask(region, width: int, height: int) -> np.ndarray:
    if region is None or isinstance(region, Special):
        return np.zeros((height, width), dtype=bool)
    if not isinstance(region, Region) or not hasattr(region, "rasterize"):
        raise TypeError("VOS measures require regions that can be rasterized")
    rendered = region.rasterize((0, 0, width - 1, height - 1))
    return np.asarray(rendered, dtype=bool)


def _iou(groundtruth: np.ndarray, prediction: np.ndarray) -> float:
    intersection = np.logical_and(groundtruth, prediction).sum()
    union = np.logical_or(groundtruth, prediction).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def _boundary(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = _morphology(mask, 1, dilate=False, disk=False)
    return np.logical_xor(mask, eroded)


def _morphology(mask: np.ndarray, radius: int, dilate: bool, disk: bool = True) -> np.ndarray:
    """Apply binary dilation or erosion using a square or disk footprint."""
    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=not dilate)
    result = np.zeros_like(mask, dtype=bool) if dilate else np.ones_like(mask, dtype=bool)
    for y in range(-radius, radius + 1):
        for x in range(-radius, radius + 1):
            if disk and x * x + y * y > radius * radius:
                continue
            view = padded[radius + y:radius + y + height, radius + x:radius + x + width]
            if dilate:
                result |= view
            else:
                result &= view
    return result


def _contour_f(groundtruth: np.ndarray, prediction: np.ndarray, tolerance: int) -> float:
    gt_boundary = _boundary(groundtruth)
    pred_boundary = _boundary(prediction)
    gt_count = int(gt_boundary.sum())
    pred_count = int(pred_boundary.sum())

    if gt_count == 0 and pred_count == 0:
        return 1.0
    if gt_count == 0 or pred_count == 0:
        return 0.0

    matched_gt = np.logical_and(gt_boundary, _morphology(pred_boundary, tolerance, dilate=True)).sum()
    matched_pred = np.logical_and(pred_boundary, _morphology(gt_boundary, tolerance, dilate=True)).sum()
    recall = float(matched_gt) / gt_count
    precision = float(matched_pred) / pred_count
    if recall + precision == 0:
        return 0.0
    return 2.0 * recall * precision / (recall + precision)


def compute_j(groundtruth: Sequence[Region], prediction: Sequence[Region],
              bounds: Tuple[int, int] = None) -> float:
    """Compute mean region similarity (J), i.e. intersection over union."""
    _validate_inputs(groundtruth, prediction)
    scores = []
    for gt_region, pred_region in zip(groundtruth, prediction):
        if isinstance(gt_region, Special):
            continue
        if gt_region is None:
            continue
        if isinstance(pred_region, Special):
            pred_region = None
        if bounds is None:
            scores.append(calculate_overlap(gt_region, pred_region) if pred_region is not None else 0.0)
        else:
            width, height = bounds
            scores.append(_iou(_mask(gt_region, width, height), _mask(pred_region, width, height)))
    return float(np.mean(scores)) if scores else 0.0


def compute_f(groundtruth: Sequence[Region], prediction: Sequence[Region],
              bounds: Tuple[int, int]) -> float:
    """Compute mean contour accuracy (F) using the DAVIS tolerance rule."""
    _validate_inputs(groundtruth, prediction)
    if bounds is None:
        raise ValueError("Image bounds are required to compute contour accuracy")
    width, height = bounds
    tolerance = max(1, int(ceil(0.008 * sqrt(width * width + height * height))))
    scores = []
    for gt_region, pred_region in zip(groundtruth, prediction):
        if isinstance(gt_region, Special):
            continue
        if gt_region is None:
            continue
        scores.append(_contour_f(_mask(gt_region, width, height), _mask(pred_region, width, height), tolerance))
    return float(np.mean(scores)) if scores else 0.0


def compute_jf(groundtruth: Sequence[Region], prediction: Sequence[Region],
               bounds: Tuple[int, int]) -> float:
    """Compute J&F as the arithmetic mean of J and F."""
    return (compute_j(groundtruth, prediction, bounds) +
            compute_f(groundtruth, prediction, bounds)) / 2.0


def _object_scores(metric, predictions, sequence, bounds):
    scores = []
    for object_id in sequence.objects():
        gt = [sequence.object(object_id, frame) for frame in range(len(sequence))]
        pred = [frame.get(object_id) for frame in predictions]
        scores.append(metric(gt, pred, bounds))
    return scores


class _VOSAnalysis(SeparableAnalysis):
    """Base class for VOS measures over multi-object experiment runs."""

    bounded = Boolean(default=True, description="Rasterize regions to the sequence canvas.")

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    def subcompute(self, experiment: Experiment, tracker: Tracker,
                   sequence: DatasetSequence, dependencies: List[Grid]):
        assert isinstance(experiment, MultiRunExperiment)
        runs = _gather_multiobject_runs(experiment, tracker, sequence)
        bounds = sequence.size if self.bounded else None
        if bounds is None and self.requires_bounds:
            raise ValueError("VOS contour measures require bounded sequences")
        values = []
        for predictions in runs:
            values.extend(_object_scores(self._compute, predictions, sequence, bounds))
        if not values:
            raise MissingResultsException("Sequence contains no objects")
        return float(np.mean(values)),

    @property
    def requires_bounds(self):
        return False

    def _compute(self, groundtruth, prediction, bounds):
        raise NotImplementedError()


class J(_VOSAnalysis):
    """Per-sequence VOS region similarity analysis."""

    @property
    def _title_default(self):
        return "Region similarity"

    def describe(self):
        return Measure(self.title, "J", minimal=0, maximal=1,
                       direction=Sorting.DESCENDING),

    def _compute(self, groundtruth, prediction, bounds):
        return compute_j(groundtruth, prediction, bounds)


class F(_VOSAnalysis):
    """Per-sequence VOS contour accuracy analysis."""

    @property
    def _title_default(self):
        return "Contour accuracy"

    def describe(self):
        return Measure(self.title, "F", minimal=0, maximal=1,
                       direction=Sorting.DESCENDING),

    @property
    def requires_bounds(self):
        return True

    def _compute(self, groundtruth, prediction, bounds):
        return compute_f(groundtruth, prediction, bounds)


class JF(_VOSAnalysis):
    """Per-sequence VOS combined J&F analysis."""

    @property
    def _title_default(self):
        return "Region and contour accuracy"

    def describe(self):
        return Measure(self.title, "J&F", minimal=0, maximal=1,
                       direction=Sorting.DESCENDING),

    @property
    def requires_bounds(self):
        return True

    def _compute(self, groundtruth, prediction, bounds):
        return compute_jf(groundtruth, prediction, bounds)


__all__ = ["compute_j", "compute_f", "compute_jf", "J", "F", "JF"]
