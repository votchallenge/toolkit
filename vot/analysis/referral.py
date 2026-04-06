"""Referral tracking accuracy analysis.

Computes mean IoU between each prompt's predicted masks and the single GT
object, averaged across prompts and then across sequences.
"""

from typing import List, Tuple, Any

import numpy as np

from attributee import Boolean, Integer, Float, Include, String

from vot.analysis import (Measure, MissingResultsException, SequenceAggregator,
                          Sorting, SeparableAnalysis, Curve, is_special)
from vot.dataset import Sequence
from vot.experiment import Experiment
from vot.experiment.referral import ReferralExperiment
from vot.region import Region, calculate_overlaps
from vot.tracker import Tracker, Trajectory
from vot.utilities.data import Grid
from vot.analysis.accuracy import gather_overlaps


class ReferralSequenceAccuracy(SeparableAnalysis):
    """Per-sequence accuracy for referral experiments.

    For each prompt, computes the mean IoU of the predicted trajectory against
    the single GT object (filtered by evaluation tag, with ignore masks).
    The sequence-level score is the mean across all prompts.
    """

    burnin = Integer(default=0, val_min=0)
    ignore_unknown = Boolean(default=False, description="If False, missing predictions count as 0 IoU.")
    ignore_invisible = Boolean(default=False)
    bounded = Boolean(default=True)
    ignore_masks = String(default="_ignore")
    filter_tag = String(default=None)

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

        bounds = sequence.size if self.bounded else None

        objects = [o for o in sequence.objects() if not o.startswith("_")]
        assert len(objects) == 1, f"Referral expects single-object sequences, got {objects}"
        groundtruth = sequence.object(objects[0])

        ignore_regions = sequence.object(self.ignore_masks)

        if self.filter_tag is not None:
            frame_mask = [self.filter_tag in sequence.tags(i)
                          for i in range(len(sequence))]
            if not any(frame_mask):
                frame_mask = None
        else:
            frame_mask = None

        prompt_accuracies = []

        for trajectory in trajectories:
            if frame_mask is not None:
                traj_f = [r for r, m in zip(trajectory, frame_mask) if m]
                gt_f = [r for r, m in zip(groundtruth, frame_mask) if m]
                masks_f = ([r for r, m in zip(ignore_regions, frame_mask) if m]
                           if ignore_regions else None)
            else:
                traj_f = list(trajectory)
                gt_f = groundtruth
                masks_f = ignore_regions

            overlaps, _ = gather_overlaps(
                traj_f, gt_f,
                burnin=self.burnin,
                ignore_unknown=self.ignore_unknown,
                ignore_invisible=self.ignore_invisible,
                bounds=bounds,
                ignore_masks=masks_f,
            )

            if overlaps.size > 0:
                prompt_accuracies.append(float(np.mean(overlaps)))
            else:
                prompt_accuracies.append(0.0)

        return float(np.mean(prompt_accuracies)),


class ReferralAverageAccuracy(SequenceAggregator):
    """Average referral accuracy across sequences."""

    analysis = Include(ReferralSequenceAccuracy)
    weighted = Boolean(default=True, description="Weight accuracy by sequence length.")

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

        return (accuracy / frames,) if frames > 0 else (0.0,)


class ReferralSuccessPlot(SeparableAnalysis):
    """Success plot for referral experiments — IoU threshold sweep."""

    ignore_unknown = Boolean(default=False)
    ignore_invisible = Boolean(default=False)
    burnin = Integer(default=0, val_min=0)
    bounded = Boolean(default=True)
    resolution = Integer(default=100, val_min=2)
    ignore_masks = String(default="_ignore")
    filter_tag = String(default=None)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, ReferralExperiment)

    @property
    def _title_default(self):
        return "Referral success plot"

    def describe(self):
        return Curve("Plot", 2, "S", minimal=(0, 0), maximal=(1, 1),
                     labels=("Threshold", "Success"), trait="success"),

    def subcompute(self, experiment: Experiment, tracker: Tracker,
                   sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:
        assert isinstance(experiment, ReferralExperiment)

        trajectories = experiment.gather(tracker, sequence)
        if len(trajectories) == 0:
            raise MissingResultsException()

        bounds = sequence.size if self.bounded else None

        objects = [o for o in sequence.objects() if not o.startswith("_")]
        assert len(objects) == 1
        groundtruth = sequence.object(objects[0])
        ignore_regions = sequence.object(self.ignore_masks)

        if self.filter_tag is not None:
            frame_mask = [self.filter_tag in sequence.tags(i)
                          for i in range(len(sequence))]
            if not any(frame_mask):
                frame_mask = None
        else:
            frame_mask = None

        axis_x = np.linspace(0, 1, self.resolution)
        axis_y = np.zeros_like(axis_x)
        valid_prompts = 0

        for trajectory in trajectories:
            if frame_mask is not None:
                traj_f = [r for r, m in zip(trajectory, frame_mask) if m]
                gt_f = [r for r, m in zip(groundtruth, frame_mask) if m]
                masks_f = ([r for r, m in zip(ignore_regions, frame_mask) if m]
                           if ignore_regions else None)
            else:
                traj_f = list(trajectory)
                gt_f = groundtruth
                masks_f = ignore_regions

            overlaps, _ = gather_overlaps(
                traj_f, gt_f,
                burnin=self.burnin,
                ignore_unknown=self.ignore_unknown,
                ignore_invisible=self.ignore_invisible,
                bounds=bounds,
                ignore_masks=masks_f,
            )

            if len(overlaps) == 0:
                continue

            valid_prompts += 1
            for i, threshold in enumerate(axis_x):
                if threshold == 1:
                    axis_y[i] += np.sum(overlaps >= threshold) / len(overlaps)
                else:
                    axis_y[i] += np.sum(overlaps > threshold) / len(overlaps)

        if valid_prompts > 0:
            axis_y /= valid_prompts

        return [(x, y) for x, y in zip(axis_x, axis_y)],


class ReferralAverageSuccessPlot(SequenceAggregator):
    """Average success plot across sequences for referral experiments."""

    resolution = Integer(default=100, val_min=2)
    analysis = Include(ReferralSuccessPlot)

    def dependencies(self):
        return self.analysis,

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, ReferralExperiment)

    @property
    def _title_default(self):
        return "Referral success plot"

    def describe(self):
        return Curve("Plot", 2, "S", minimal=(0, 0), maximal=(1, 1),
                     labels=("Threshold", "Success"), trait="success"),

    def aggregate(self, _: Tracker, sequences: List[Sequence],
                  results: Grid):
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
