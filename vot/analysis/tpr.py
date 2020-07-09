import math
import numpy as np
from typing import List

from vot.tracker import Tracker
from vot.dataset import Sequence
from vot.region import Region, RegionType, calculate_overlaps
from vot.experiment import Experiment
from vot.experiment.multirun import UnsupervisedExperiment
from vot.analysis import TrackerSeparableAnalysis, DependentAnalysis, \
    MissingResultsException, Measure, Sorting, Curve, Plot, Axis
from vot.utilities import alias
from vot.utilities.attributes import Float, Integer, Boolean, Include

def determine_thresholds(scores: List[float], resolution: int) -> List[float]:
    scores = [score for score in scores if not math.isnan(score) ] #and not score is None]
    scores = sorted(scores, reverse=True)

    if len(scores) > resolution - 2:
        delta = math.floor(len(scores) / (resolution - 2))
        idxs = np.round(np.linspace(delta, len(scores) - delta, num=resolution - 2)).astype(np.int)
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

    for i, threshold in enumerate(thresholds):

        subset = confidence >= threshold

        if np.sum(subset) == 0:
            precision[i] = 1
            recall[i] = 0
        else:
            precision[i] = np.mean(overlaps[subset])
            recall[i] = np.sum(overlaps[subset]) / n_visible

    return precision, recall

@alias("Tracking Precision/Recall Curve", "prcurve" ,"PrecisionRecallCurve")
class PrecisionRecallCurve(TrackerSeparableAnalysis):

    resolution = Integer(default=100)
    ignore_unknown = Boolean(default=True)
    bounded = Boolean(default=True)

    @property
    def title(self):
        return "Tracking precision/recall"

    def describe(self):
        return Curve("Precision Recall curve", dimensions=2, abbreviation="PR", minimal=(0, 0), maximal=(1, 1), labels=("Recall", "Precision")), None

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, UnsupervisedExperiment)

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequences: List[Sequence]):

        # calculate thresholds
        total_scores = 0
        for sequence in sequences:
            trajectories = experiment.gather(tracker, sequence)
    
            if len(trajectories) == 0:
                raise MissingResultsException("Missing results for sequence {}".format(sequence.name))

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

        thresholds = determine_thresholds(scores_all, self.resolution)

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
                pr_, re_ = compute_tpr_curves(trajectory.regions(), conf_, sequence, thresholds, self.ignore_unknown, self.bounded)
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

        curve = [(re / len(pr_curves), pr / len(pr_curves)) for pr, re in zip(pr_curve, re_curve)]

        return curve, thresholds

@alias("Tracking F-Score Curve", "FScoreCurve", "fcurve", "fscorecurve")
class FScoreCurve(DependentAnalysis):

    beta = Float(default=1)
    prcurve = Include(PrecisionRecallCurve)

    @property
    def title(self):
        return "Tracking precision/recall"

    def describe(self):
        return Plot("Tracking F-score curve", "F", wrt="normalized threshold", minimal=0, maximal=1), None

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, UnsupervisedExperiment)

    def dependencies(self):
        return self.prcurve,

    def join(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence], results):
        processed_results = []

        for result in results[0]:
            beta2 = (self.beta * self.beta)
            f_curve = [((1 + beta2) * pr_ * re_) / (beta2 * pr_ + re_) for pr_, re_ in result[0]]

            processed_results.append((f_curve, result[0][1]))

        return processed_results

    def axes(self):
        return Axis.TRACKERS,

@alias("Best Tracking Precision/Recall based on FScore", "PrecisionRecall", "tpr")
class PrecisionRecall(DependentAnalysis):

    prcurve = Include(PrecisionRecallCurve)
    fcurve = Include(FScoreCurve)

    @property
    def title(self):
        return "Tracking precision/recall"

    def describe(self):
        return Measure("Precision", "Pr", minimal=0, maximal=1, direction=Sorting.DESCENDING), \
             Measure("Recall", "Re", minimal=0, maximal=1, direction=Sorting.DESCENDING), \
             Measure("F Score", "F", minimal=0, maximal=1, direction=Sorting.DESCENDING)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, UnsupervisedExperiment)

    def dependencies(self):
        return self.prcurve, self.fcurve

    def join(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence], results):

        f_curves = results[1]
        pr_curves = results[0]

        joined = []

        for f_curve, pr_curve in zip(f_curves, pr_curves):
            # get optimal F-score and Pr and Re at this threshold
            f_score = max(f_curve[0])
            best_i = f_curve[0].index(f_score)
            re_score = pr_curve[0][best_i][0]
            pr_score = pr_curve[0][best_i][1]
            joined.append((pr_score, re_score, f_score))

        return joined

    def axes(self):
        return Axis.TRACKERS,