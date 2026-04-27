"""Referral tracking accuracy analysis.

Computes mean IoU between each prompt's predicted masks and the single GT
object, averaged across prompts and then across sequences.

Per-frame IoU semantics (matches the participant's evaluation pipeline):
  - "0" line in groundtruth_object1.txt -> frame has no annotation (6fps gap):
    SKIP (does not contribute to the average).
  - "m..." line whose mask is empty -> frame is annotated and the GT mask is
    empty (e.g. the referenced object disappeared): COUNT.
        * if prediction is also empty -> IoU = 1
        * if prediction is non-empty  -> IoU = 0
  - "m..." line whose mask is non-empty -> standard IoU vs prediction
    (with ignore mask applied if present).
"""

from typing import List, Tuple, Any

import numpy as np

from attributee import Boolean, Include, String

from vot.analysis import (Measure, MissingResultsException, SequenceAggregator,
                          Sorting, SeparableAnalysis, is_special)
from vot.dataset import Sequence
from vot.experiment import Experiment
from vot.experiment.referral import ReferralExperiment
from vot.region.shapes import Mask
from vot.tracker import Tracker
from vot.utilities.data import Grid


def _render_mask(region, height: int, width: int):
    """Render a Region into a (H, W) bool numpy array. Returns None if the
    region cannot be interpreted as a mask (unknown / non-mask shape).
    """
    if region is None:
        return None
    if not isinstance(region, Mask):
        return None
    m = region.mask
    if m.size == 0:
        return np.zeros((height, width), dtype=bool)
    ox, oy = region.offset
    out = np.zeros((height, width), dtype=bool)
    h0, w0 = m.shape
    x0, y0 = max(ox, 0), max(oy, 0)
    x1, y1 = min(ox + w0, width), min(oy + h0, height)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = m[y0 - oy:y1 - oy, x0 - ox:x1 - ox].astype(bool)
    return out


def _per_frame_iou(gt_mask, pred_mask, ignore_mask=None) -> float:
    """IoU between two (H, W) bool arrays, applying ignore mask.

    Empty-mask convention:
      both empty -> 1.0
      gt empty, pred non-empty -> 0.0
      gt non-empty, pred empty -> 0.0
      otherwise -> standard IoU (intersection / union).
    """
    if ignore_mask is not None:
        gt_mask = gt_mask & ~ignore_mask
        pred_mask = pred_mask & ~ignore_mask

    gt_any = bool(gt_mask.any())
    pr_any = bool(pred_mask.any())

    if not gt_any and not pr_any:
        return 1.0
    if not gt_any or not pr_any:
        return 0.0
    inter = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return float(inter) / float(union) if union > 0 else 1.0


class ReferralSequenceAccuracy(SeparableAnalysis):
    """Per-sequence accuracy for referral experiments.

    Collects IoU values for every (prompt, frame) pair that has an annotation
    (skipping the 6fps gaps where groundtruth_object1.txt has "0"). For
    annotated frames whose GT mask is empty (e.g. the object disappeared)
    the IoU is 1 if the prediction is also empty, else 0.
    """

    ignore_masks = String(default="_ignore",
        description="Object id whose regions are treated as ignore masks.")
    filter_tag = String(default=None,
        description="Only count frames carrying this tag.")

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, ReferralExperiment)

    @property
    def _title_default(self):
        return "Referral sequence accuracy"

    def describe(self):
        return Measure(self.title, "", 0, 1, Sorting.DESCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker,
                   sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:
        assert isinstance(experiment, ReferralExperiment)

        trajectories = experiment.gather(tracker, sequence)
        if len(trajectories) == 0:
            raise MissingResultsException()

        objects = [o for o in sequence.objects() if not o.startswith("_")]
        assert len(objects) == 1, f"Referral expects single-object sequences, got {objects}"
        groundtruth = sequence.object(objects[0])
        ignore_regions = sequence.object(self.ignore_masks)

        W, H = sequence.size

        if self.filter_tag is not None:
            frame_mask = [self.filter_tag in sequence.tags(i)
                          for i in range(len(sequence))]
            if not any(frame_mask):
                frame_mask = None
        else:
            frame_mask = None

        # Pre-render GT and ignore masks once (shared across prompts).
        gt_arrays: List = []
        annotated: List[bool] = []
        for i, gt in enumerate(groundtruth):
            if is_special(gt, Sequence.UNKNOWN):
                gt_arrays.append(None); annotated.append(False); continue
            if frame_mask is not None and not frame_mask[i]:
                gt_arrays.append(None); annotated.append(False); continue
            gt_arrays.append(_render_mask(gt, H, W))
            annotated.append(gt_arrays[-1] is not None)

        ig_arrays = [None] * len(groundtruth)
        if ignore_regions is not None:
            for i, ig in enumerate(ignore_regions):
                if isinstance(ig, Mask):
                    ig_arrays[i] = _render_mask(ig, H, W)

        all_overlaps: List[float] = []

        for trajectory in trajectories:
            pred_regions = trajectory.regions()
            for i in range(len(groundtruth)):
                if not annotated[i]:
                    continue
                gm = gt_arrays[i]
                pred_region = pred_regions[i] if i < len(pred_regions) else None
                pm = _render_mask(pred_region, H, W)
                if pm is None:
                    pm = np.zeros((H, W), dtype=bool)
                all_overlaps.append(_per_frame_iou(gm, pm, ig_arrays[i]))

        return (float(np.mean(all_overlaps)) if all_overlaps else 0.0),


class ReferralAverageAccuracy(SequenceAggregator):
    """Average referral accuracy across sequences.

    Reports the mean IoU over every evaluated (sequence, prompt, frame) triple,
    weighted by the number of evaluated frames per sequence (i.e. true mean
    IoU per predicted frame). Frames without GT (6fps gaps, empty GT masks)
    are excluded by ReferralSequenceAccuracy.
    """

    analysis = Include(ReferralSequenceAccuracy)
    weighted = Boolean(default=True,
        description="Weight by sequence length (proportional to number of evaluated frames).")

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, ReferralExperiment)

    @property
    def _title_default(self):
        return "Referral accuracy"

    def dependencies(self):
        return self.analysis,

    def describe(self):
        return Measure(self.title, "", 0, 1, Sorting.DESCENDING),

    def aggregate(self, _: Tracker, sequences: List[Sequence],
                  results: Grid):
        accuracy = 0.0
        weight = 0.0

        for i, sequence in enumerate(sequences):
            if results[i, 0] is None:
                continue
            seq_iou = results[i, 0][0]

            if self.weighted:
                accuracy += seq_iou * len(sequence)
                weight += len(sequence)
            else:
                accuracy += seq_iou
                weight += 1

        return (accuracy / weight,) if weight > 0 else (0.0,)


