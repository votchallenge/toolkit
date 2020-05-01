import math
import numpy as np
from typing import List, Dict, Any

from vot.tracker import Tracker
from vot.dataset import Sequence
from vot.region import Region, RegionType, calculate_overlaps
from vot.experiment import Experiment
from vot.experiment.multirun import UnsupervisedExperiment
from vot.analysis import Analysis, DependentAnalysis, MissingResultsException, Curve
from vot.utilities import to_number, to_logical

def determine_thresholds(scores: List[float], resolution: int) -> List[float]:
    scores = [score for score in scores if not math.isnan(score)]
    scores = sorted(scores, reverse=True)

    if len(scores) > resolution - 2:
        delta = math.floor(len(scores) / (resolution - 2))
        idxs = np.round(np.linspace(delta, len(scores) - delta, num = resolution - 2)).astype(np.int)
        thresholds = [scores[idx] for idx in idxs]
    else:
        thresholds = scores

    thresholds.insert(0, math.inf)
    thresholds.insert(len(thresholds), -math.inf)

    return thresholds

def compute_tpr_curves(trajectory: List[Region], confidence: List[float], sequence: Sequence, thresholds: List[float],
    ignore_unknown: bool = True, bounded: bool = True):

    overlaps = np.array(calculate_overlaps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    confidence = np.array(confidence)

    n_visible = len([region for region in sequence.groundtruth() if region.type is not RegionType.SPECIAL])

    precision = len(thresholds) * [float(0)]
    recall = len(thresholds) * [float(0)]

    for i in range(len(thresholds)):

        subset = confidence >= thresholds[i]

        if np.sum(subset) == 0:
            precision[i] = 1
            recall[i] = 0
        else:
            precision[i] = np.mean(overlaps[subset])
            recall[i] = np.sum(overlaps[subset]) / n_visible

    return precision, recall

class PrecisionRecallCurve(Analysis):

    def __init__(self, resolution: int = 100, ignore_unknown: bool = True, bounded: bool = True):
        super().__init__()
        self._resolution = to_number(resolution, min_n=1)
        self._ignore_unknown = to_logical(ignore_unknown)
        self._bounded = to_logical(bounded)

    @property
    def name(self):
        return "Tracking precision/recall"

    def parameters(self) -> Dict[str, Any]:
        return dict(resolution=self._resolution, ignore_unknown=self._ignore_unknown, bounded=self._bounded)

    def describe(self):
        return Curve("Precision Recall curve", dimensions=2, abbreviation="PR", minimal=(0, 0), maximal=(1, 1)),

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, UnsupervisedExperiment)

    def compute(self, tracker: Tracker, experiment: Experiment, sequences: List[Sequence]):

        # calculate thresholds
        total_scores = 0
        for sequence in sequences:
            trajectories = experiment.gather(tracker, sequence)
            for trajectory in trajectories:
                total_scores += len(trajectory)

        # allocate memory for all scores
        scores_all = total_scores * [float(0)]

        idx = 0
        for sequence in sequences:
            trajectories = experiment.gather(tracker, sequence)
            for trajectory in trajectories:
                conf_ = [trajectory.properties(i).get('confidence', 0) for i in range(len(trajectory))]
                scores_all[idx:idx + len(conf_)] = conf_
                idx += len(conf_)

        thresholds = determine_thresholds(scores_all, self._resolution)

        # calculate per-sequence Precision and Recall curves
        pr_curves = []
        re_curves = []

        for sequence in sequences:

            trajectories = experiment.gather(tracker, sequence)

            if len(trajectories) == 0:
                raise MissingResultsException()

            pr = len(thresholds) * [float(0)]
            re = len(thresholds) * [float(0)]
            for trajectory in trajectories:
                conf_ = [trajectory.properties(i).get('confidence', 0) for i in range(len(trajectory))]
                pr_, re_ = compute_tpr_curves(trajectory.regions(), conf_, sequence, thresholds, self._ignore_unknown, self._bounded)
                pr = [p1 + p2 for p1, p2 in zip(pr, pr_)]
                re = [r1 + r2 for r1, r2 in zip(re, re_)]

            pr = [p1 / len(trajectories) for p1 in pr]
            re = [r1 / len(trajectories) for r1 in re]

            pr_curves.append(pr)
            re_curves.append(re)

        # calculate a single Precision, Recall and F-score curves for a given tracker
        # average Pr-Re curves over the sequences
        pr_curve = len(thresholds) * [float(0)]
        re_curve = len(thresholds) * [float(0)]

        for i, _ in enumerate(thresholds):
            for j, _ in enumerate(pr_curves):
                pr_curve[i] += pr_curves[j][i]
                re_curve[i] += re_curves[j][i]

        curve = [(pr / len(pr_curves), re / len(pr_curves)) for pr, re in zip(pr_curve, re_curve)]

        return curve

class FScoreCurve(DependentAnalysis):

    def __init__(self, resolution: int = 100, ignore_unknown: bool = True, bounded: bool = True):
        super().__init__()
        self._resolution = resolution
        self._ignore_unknown = ignore_unknown
        self._bounded = bounded
        self._prcurve = PrecisionRecallCurve(resolution, ignore_unknown, bounded)

    @property
    def name(self):
        return "Tracking precision/recall"

    def parameters(self) -> Dict[str, Any]:
        return dict(resolution=self._resolution, ignore_unknown=self._ignore_unknown, bounded=self._bounded)

    def describe(self):
        return Curve("F", None),

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, UnsupervisedExperiment)

    def dependencies(self):
        return [self._prcurve]

    def join(self, results):

        f_curve = [(2 * pr_ * re_) / (pr_ + re_) for pr_, re_ in results[0][0]]

        return f_curve,

