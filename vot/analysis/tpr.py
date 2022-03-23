import math
import numpy as np
from typing import List, Iterable, Tuple, Any
import itertools

from attributee import Float, Integer, Boolean, Include

from vot.tracker import Tracker
from vot.dataset import Sequence
from vot.region import Region, RegionType, calculate_overlaps
from vot.experiment import Experiment
from vot.experiment.multirun import UnsupervisedExperiment
from vot.analysis import SequenceAggregator, Analysis, SeparableAnalysis, \
    MissingResultsException, Measure, Sorting, Curve, Plot, SequenceAggregator, \
    Axes, analysis_registry
from vot.utilities.data import Grid

def determine_thresholds(scores: Iterable[float], resolution: int) -> List[float]:
    scores = [score for score in scores if not math.isnan(score)] #and not score is None]
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

class _ConfidenceScores(SeparableAnalysis):

    @property
    def title(self):
        return "Aggregate confidence scores"

    def describe(self):
        return None,

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, UnsupervisedExperiment)

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:

        scores_all = []
        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException("Missing results for sequence {}".format(sequence.name))

        for trajectory in trajectories:
            confidence = [trajectory.properties(i).get('confidence', 0) for i in range(len(trajectory))]
            scores_all.extend(confidence)

        return scores_all,


class _Thresholds(SequenceAggregator):

    resolution = Integer(default=100)

    @property
    def title(self):
        return "Thresholds for tracking precision/recall"

    def describe(self):
        return None,

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, UnsupervisedExperiment)

    def dependencies(self):
        return _ConfidenceScores(),

    def aggregate(self, tracker: Tracker, sequences: List[Sequence], results: Grid) -> Tuple[Any]:

        thresholds = determine_thresholds(itertools.chain(*[result[0] for result in results]), self.resolution),

        return thresholds,

@analysis_registry.register("pr_curves")
class PrecisionRecallCurves(SeparableAnalysis):

    thresholds = Include(_Thresholds)
    ignore_unknown = Boolean(default=True)
    bounded = Boolean(default=True)

    @property
    def title(self):
        return "Tracking precision/recall"

    def describe(self):
        return Curve("Precision Recall curve", dimensions=2, abbreviation="PR", minimal=(0, 0), maximal=(1, 1), labels=("Recall", "Precision")), None

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, UnsupervisedExperiment)

    def dependencies(self):
        return self.thresholds,

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:

        thresholds = dependencies[0, 0][0][0] # dependencies[0][0, 0]

        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        precision = len(thresholds) * [float(0)]
        recall = len(thresholds) * [float(0)]
        for trajectory in trajectories:
            confidence = [trajectory.properties(i).get('confidence', 0) for i in range(len(trajectory))]
            pr, re = compute_tpr_curves(trajectory.regions(), confidence, sequence, thresholds, self.ignore_unknown, self.bounded)
            for i in range(len(thresholds)):
                precision[i] += pr[i]
                recall[i] += re[i]

#         return [(re / len(trajectories), pr / len(trajectories)) for pr, re in zip(precision, recall)], thresholds
        return [(pr / len(trajectories), re / len(trajectories)) for pr, re in zip(precision, recall)], thresholds

@analysis_registry.register("pr_curve")
class PrecisionRecallCurve(SequenceAggregator):

    curves = Include(PrecisionRecallCurves)

    @property
    def title(self):
        return "Tracking precision/recall average curve"

    def describe(self):
        return self.curves.describe()

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, UnsupervisedExperiment)

    def dependencies(self):
        return self.curves,

    # def collapse(self, tracker: Tracker, sequences: List[Sequence], results: Grid) -> Tuple[Any]:
    def aggregate(self, tracker: Tracker, sequences: List[Sequence], results: Grid) -> Tuple[Any]:

        curve = None
        thresholds = None

        for partial, thresholds in results:
            if curve is None:
                curve = partial
                continue

            curve = [(pr1 + pr2, re1 + re2) for (pr1, re1), (pr2, re2) in zip(curve, partial)]

        curve = [(re / len(results), pr / len(results)) for pr, re in curve]

        return curve, thresholds


@analysis_registry.register("f_curve")
class FScoreCurve(Analysis):

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

    def compute(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence], dependencies: List[Grid]) -> Grid:
        processed_results = Grid(len(trackers), 1)

        for i, result in enumerate(dependencies[0]):
            beta2 = (self.beta * self.beta)
            f_curve = [((1 + beta2) * pr_ * re_) / (beta2 * pr_ + re_) for pr_, re_ in result[0]]

            processed_results[i, 0] = (f_curve, result[0][1])

        return processed_results

    @property
    def axes(self):
        return Axes.TRACKERS

@analysis_registry.register("average_tpr")
class PrecisionRecall(Analysis):

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

    def compute(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence], dependencies: List[Grid]) -> Grid:

        f_curves = dependencies[1]
        pr_curves = dependencies[0]

        joined = Grid(len(trackers), 1)

        for i, (f_curve, pr_curve) in enumerate(zip(f_curves, pr_curves)):
            # get optimal F-score and Pr and Re at this threshold
            f_score = max(f_curve[0])
            best_i = f_curve[0].index(f_score)
            re_score = pr_curve[0][best_i][0]
            pr_score = pr_curve[0][best_i][1]
            joined[i, 0] = (pr_score, re_score, f_score)

        return joined

    @property
    def axes(self):
        return Axes.TRACKERS
