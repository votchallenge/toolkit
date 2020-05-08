import math
from typing import List, Dict, Any, Tuple
from collections import Counter

import numpy as np

from vot.tracker import Tracker, Trajectory
from vot.dataset import Sequence
from vot.dataset.proxy import FrameMapSequence
from vot.experiment import Experiment
from vot.experiment.multirun import MultiRunExperiment, SupervisedExperiment, UnsupervisedExperiment
from vot.experiment.multistart import MultiStartExperiment, find_anchors
from vot.region import Region, Special, calculate_overlaps
from vot.analysis import SeparatableAnalysis, DependentAnalysis, \
    MissingResultsException, Measure, Point, is_special
from vot.analysis.curves import PrecisionRecallCurve, FScoreCurve
from vot.utilities import to_number, to_logical

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

def compute_eao_partial(overlaps: List, success: List[bool], curve_length: int):
    phi = curve_length * [float(0)]
    active = curve_length * [float(0)]

    for i, (o, success) in enumerate(zip(overlaps, success)):

        o_array = np.array(o)

        for j in range(1, curve_length):

            if j < len(o):
                phi[j] += np.mean(o_array[1:j+1])
                active[j] += 1
            elif not success:
                phi[j] += np.sum(o_array[1:len(o)]) / (j - 1)
                active[j] += 1

    phi = [p / a if a > 0 else 0 for p, a in zip(phi, active)]
    return phi, active

def count_failures(trajectory: List[Region]) -> Tuple[int, int]:
    return len([region for region in trajectory if is_special(region, Special.FAILURE)]), len(trajectory)

class AverageAccuracy(SeparatableAnalysis):

    def __init__(self, burnin: int = 10, ignore_unknown: bool = True, bounded: bool = True):
        super().__init__()
        self._burnin = to_number(burnin, min_n=0)
        self._ignore_unknown = to_logical(ignore_unknown)
        self._bounded = to_logical(bounded)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def name(self):
        return "Average accurarcy"

    def parameters(self) -> Dict[str, Any]:
        return dict(burnin=self._burnin, ignore_unknown=self._ignore_unknown, bounded=self._bounded)

    def describe(self):
        return Measure("Accuracy", "AUC", 0, 1, Measure.DESCENDING), \
            None

    # TODO: turn off weighted average
    def join(self, results: List[tuple]):
        accuracy = 0
        frames = 0

        for a, n in results:
            accuracy = accuracy + a * n
            frames = frames + n

        return accuracy / frames, frames

    def compute_partial(self, tracker: Tracker, experiment: Experiment, sequence: Sequence):

        if isinstance(experiment, MultiRunExperiment):
            trajectories = experiment.gather(tracker, sequence)

            if len(trajectories) == 0:
                raise MissingResultsException()

            cummulative = 0
            for trajectory in trajectories:
                accuracy, _ = compute_accuracy(trajectory.regions(), sequence, self._burnin, self._ignore_unknown, self._bounded)
                cummulative = cummulative + accuracy

            return cummulative / len(trajectories), len(sequence)

class FailureCount(SeparatableAnalysis):

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, SupervisedExperiment)

    @property
    def name(self):
        return "Number of failures"

    def describe(self):
        return Measure("Failures", "F", 0, None, Measure.ASCENDING), \
            None

    def join(self, results: List[tuple]):
        failures = 0
        frames = 0

        for a, n in results:
            failures = failures + a
            frames = frames + n

        return failures, frames

    def compute_partial(self, tracker: Tracker, experiment: Experiment, sequence: Sequence):
        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        failures = 0
        for trajectory in trajectories:
            failures = failures + count_failures(trajectory.regions())[0]

        return failures / len(trajectories), len(trajectories[0])

class PrecisionRecall(DependentAnalysis):

    def __init__(self, resolution: int = 100, ignore_unknown: bool = True, bounded: bool = True):
        super().__init__()
        self._resolution = resolution
        self._ignore_unknown = ignore_unknown
        self._bounded = bounded
        self._prcurve = PrecisionRecallCurve(resolution, ignore_unknown, bounded)
        self._fcurve = FScoreCurve(resolution, ignore_unknown, bounded)

    @property
    def name(self):
        return "Tracking precision/recall"

    def parameters(self) -> Dict[str, Any]:
        return dict(resolution=self._resolution, ignore_unknown=self._ignore_unknown, bounded=self._bounded)

    def describe(self):
        return Measure("Precision", "Pr", minimal=0, maximal=1, direction=Measure.DESCENDING), \
             Measure("Recall", "Re", minimal=0, maximal=1, direction=Measure.DESCENDING), \
             Measure("F", minimal=0, maximal=1, direction=Measure.DESCENDING)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, UnsupervisedExperiment)

    def dependencies(self):
        return [self._prcurve, self._fcurve]

    def join(self, results: List[tuple]):

        # get optimal F-score and Pr and Re at this threshold
        f_score = max(results[1][0])
        best_i = results[1][0].index(f_score)
        pr_score = results[0][0][best_i][0]
        re_score = results[0][0][best_i][1]

        return pr_score, re_score, f_score

class AccuracyRobustness(SeparatableAnalysis):

    def __init__(self, sensitivity: int = 30, burnin: int = 10, ignore_unknown: bool = True, bounded: bool = True):
        super().__init__()
        self._sensitivity = to_number(sensitivity, min_n=1)
        self._burnin = to_number(burnin, min_n=0)
        self._ignore_unknown = to_logical(ignore_unknown)
        self._bounded = to_logical(bounded)

    @property
    def name(self):
        return "AR analysis"

    def parameters(self) -> Dict[str, Any]:
        return dict(sensitivity=self._sensitivity, burnin=self._burnin, ignore_unknown=self._ignore_unknown, bounded=self._bounded)

    def describe(self):
        return Measure("Accuracy", "A", minimal=0, maximal=1, direction=Measure.DESCENDING), \
             Measure("Robustness", "R", minimal=0, direction=Measure.DESCENDING), \
             Point("AR plot", dimensions=2, abbreviation="AR", minimal=(0, 0), maximal=(1, 1)), \
             None

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, SupervisedExperiment)

    def join(self, results: List[tuple]):
        failures = 0
        accuracy = 0
        weight_total = 0

        for a, f, _, w in results:
            failures += f * w
            accuracy += a * w
            weight_total += w

        ar = (accuracy / weight_total, math.exp(- (failures / weight_total) * float(self._sensitivity)))

        return accuracy / weight_total, failures / weight_total, ar, weight_total

    def compute_partial(self, tracker: Tracker, experiment: Experiment, sequence: Sequence):
        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        accuracy = 0
        failures = 0
        for trajectory in trajectories:
            failures += count_failures(trajectory.regions())[0]
            accuracy += compute_accuracy(trajectory.regions(), sequence, self._burnin, self._ignore_unknown, self._bounded)[0]

        ar = (accuracy / len(trajectories), math.exp(- (float(failures) / len(trajectories)) * float(self._sensitivity)))

        return accuracy / len(trajectories), failures / len(trajectories), ar, len(trajectories[0])

class AccuracyRobustnessMultiStart(SeparatableAnalysis):

    def __init__(self, burnin: int = 10, grace: int = 10, bounded: bool = True):
        super().__init__()
        self._burnin = burnin
        self._grace = grace
        self._bounded = bounded
        self._threshold = 0.1

    @property
    def name(self):
        return "AR analysis"

    def parameters(self) -> Dict[str, Any]:
        return dict(burnin=self._burnin, grace=self._grace, bounded=self._bounded)

    def describe(self):
        return Measure("Accuracy", "A", minimal=0, maximal=1, direction=Measure.DESCENDING), \
             Measure("Robustness", "R", minimal=0, direction=Measure.DESCENDING), \
             Point("AR plot", dimensions=2, abbreviation="AR", minimal=(0, 0), maximal=(1, 1)), \
             None, None

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiStartExperiment)

    def join(self, results: List[tuple]):
        total_accuracy = 0
        total_robustness = 0
        weight_accuracy = 0
        weight_robustness = 0

        for accuracy, robustness, _, accuracy_w, robustness_w in results:
            total_accuracy += accuracy * accuracy_w
            total_robustness += robustness * robustness_w
            weight_accuracy += accuracy_w
            weight_robustness += robustness_w

        ar = (total_accuracy / weight_accuracy, total_robustness / weight_robustness)

        return total_accuracy / weight_accuracy, total_robustness / weight_robustness, ar, weight_accuracy, weight_robustness

    def compute_partial(self, tracker: Tracker, experiment: Experiment, sequence: Sequence):

        results = experiment.results(tracker, sequence)

        forward, backward = find_anchors(sequence, experiment.anchor)

        if not forward and not backward:
            raise RuntimeError("Sequence does not contain any anchors")

        robustness = 0
        accuracy = 0
        total = 0
        for i, reverse in [(f, False) for f in forward] + [(f, True) for f in backward]:
            name = "%s_%08d" % (sequence.name, i)

            if not Trajectory.exists(results, name):
                raise MissingResultsException()

            if reverse:
                proxy = FrameMapSequence(sequence, list(reversed(range(0, i + 1))))
            else:
                proxy = FrameMapSequence(sequence, list(range(i, sequence.length)))

            trajectory = Trajectory.read(results, name)

            overlaps = calculate_overlaps(trajectory.regions(), proxy.groundtruth(), (proxy.size) if self._burnin else None)

            grace = self._grace
            progress = len(proxy)

            for j, overlap in enumerate(overlaps):
                if overlap <= self._threshold and not proxy.groundtruth(j).is_empty():
                    grace = grace - 1
                    if grace == 0:
                        progress = j + 1 - self._grace  # subtract since we need actual point of the failure
                        break
                else:
                    grace = self._grace

            robustness += progress  # simplified original equation: len(proxy) * (progress / len(proxy))
            accuracy += sum(overlaps[0:progress])
            total += len(proxy)

        ar = (accuracy / robustness if robustness > 0 else 0, robustness / total)

        return accuracy / robustness if robustness > 0 else 0, robustness / total, ar, robustness, len(sequence)

class EAOScore(DependentAnalysis):
    def __init__(self, burnin: int = 10, grace: int = 10, bounded: bool = True, low: int = 99, high: int = 355):
        from vot.analysis.plots import EAOCurve

        super().__init__()
        self._burnin = to_number(burnin, min_n=0)
        self._grace = to_number(grace, min_n=0)
        self._bounded = to_logical(bounded)
        self._low = to_number(low, min_n=0)
        self._high = to_number(high, min_n=self._low+1)
        self._eaocurve = EAOCurve(burnin, grace, bounded)

    @property
    def name(self):
        return "EAO analysis"

    def parameters(self) -> Dict[str, Any]:
        return dict(burnin=self._burnin, grace=self._grace, bounded=self._bounded, low=self._low, high=self._high)

    def describe(self):
        return Measure("EAO", 0, 1, Measure.DESCENDING),

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, SupervisedExperiment)

    def dependencies(self):
        return self._eaocurve,

    def join(self, results: List[tuple]):
        return float(np.mean(results[0][0][self._low:self._high + 1])),

class EAOScoreMultiStart(SeparatableAnalysis):
    def __init__(self, burnin: int = 10, grace: int = 10, bounded: bool = True, threshold: float = 0.1, low: int = 115, high: int = 755):
        from vot.analysis.plots import EAOCurveMultiStart
        super().__init__()
        self._burnin = to_number(burnin, min_n=0)
        self._grace = to_number(grace, min_n=0)
        self._bounded = to_logical(bounded)
        self._threshold = to_number(threshold, min_n=0, max_n=1, conversion=float)

        self._low = to_number(low, min_n=0)
        self._high = to_number(high, min_n=self._low+1)

    @property
    def name(self):
        return "EAO analysis"

    def parameters(self) -> Dict[str, Any]:
        return dict(burnin=self._burnin, grace=self._grace, bounded=self._bounded, threshold=self._threshold, low=self._low, high=self._high)

    def describe(self):
        return Measure("EAO", 0, 1, Measure.DESCENDING),

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiStartExperiment)

    def join(self, results: List[tuple]):
        eao_curve = self._high * [float(0)]
        eao_weights = self._high * [float(0)]

        for (seq_eao_curve, eao_active), seq_w in results:
            for i, (eao_, active_) in enumerate(zip(seq_eao_curve, eao_active)):
                eao_curve[i] += eao_ * active_ * seq_w
                eao_weights[i] += active_ * seq_w

        eao_curve_final = np.array([eao_ / w_ if w_ > 0 else 0 for eao_, w_ in zip(eao_curve, eao_weights)])
        return np.mean(eao_curve_final[self._low:self._high+1])

    def compute_partial(self, tracker: Tracker, experiment: Experiment, sequence: Sequence):

        results = experiment.results(tracker, sequence)

        forward, backward = find_anchors(sequence, experiment.anchor)

        if len(forward) == 0 and len(backward) == 0:
            raise RuntimeError("Sequence does not contain any anchors")

        overlaps_all = []
        success_all = []

        for i, reverse in [(f, False) for f in forward] + [(f, True) for f in backward]:
            name = "%s_%08d" % (sequence.name, i)

            if not Trajectory.exists(results, name):
                raise MissingResultsException()

            if reverse:
                proxy = FrameMapSequence(sequence, list(reversed(range(0, i + 1))))
            else:
                proxy = FrameMapSequence(sequence, list(range(i, sequence.length)))

            trajectory = Trajectory.read(results, name)

            overlaps = calculate_overlaps(trajectory.regions(), proxy.groundtruth(), proxy.size if self._burnin else None)

            grace = self._grace
            progress = len(proxy)

            for j, overlap in enumerate(overlaps):
                if overlap <= self._threshold and not proxy.groundtruth(j).is_empty():
                    grace = grace - 1
                    if grace == 0:
                        progress = j + 1 - self._grace  # subtract since we need actual point of the failure
                        break
                else:
                    grace = self._grace

            success = True
            if progress < len(overlaps):
                # tracker has failed during this run
                overlaps[progress:] = (len(overlaps) - progress) * [float(0)]
                success = False

            overlaps_all.append(overlaps)
            success_all.append(success)

        return compute_eao_partial(overlaps_all, success_all, self._high), 1

class AttributeMultiStart(SeparatableAnalysis):
    def __init__(self, burnin: int = 10, grace: int = 10, bounded: bool = True):
        super().__init__()
        self._burnin = burnin
        self._grace = grace
        self._bounded = bounded
        self._threshold = 0.1

    @property
    def name(self):
        return "AR per-attribute analysis"

    def parameters(self) -> Dict[str, Any]:
        return dict(burnin=self._burnin, grace=self._grace, bounded=self._bounded)

    def describe(self):
        return MeasureDescription("Attr. Accuracy", 0, 1, MeasureDescription.DESCENDING), \
            MeasureDescription("Attr. Robustness", 0, 1, MeasureDescription.DESCENDING), \
            None

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiStartExperiment)

    def join(self, results: List[tuple]):
        accuracy = Counter()
        robustness = Counter()
        attribute_total = Counter()

        for seq_acc, seq_rob, seq_attr_count in results:
            for t in seq_attr_count:
                accuracy[t] += (seq_acc[t] if t in seq_acc else 0) * seq_attr_count[t]
                robustness[t] += seq_rob * seq_attr_count[t]
                attribute_total[t] += seq_attr_count[t]

        for t in attribute_total:
            accuracy[t] /= attribute_total[t]
            robustness[t] /= attribute_total[t]

        return accuracy, robustness, attribute_total

    def compute_partial(self, tracker: Tracker, experiment: Experiment, sequence: Sequence):

        results = experiment.results(tracker, sequence)

        forward, backward = find_anchors(sequence, experiment.anchor)

        if len(forward) == 0 and len(backward) == 0:
            raise RuntimeError("Sequence does not contain any anchors")

        accuracy_ = Counter()
        tags_count_ = Counter()
        robustness_ = 0
        total_ = 0
        for i, reverse in [(f, False) for f in forward] + [(f, True) for f in backward]:
            name = "%s_%08d" % (sequence.name, i)

            if not Trajectory.exists(results, name):
                raise MissingResultsException()

            if reverse:
                proxy = FrameMapSequence(sequence, list(reversed(range(0, i + 1))))
            else:
                proxy = FrameMapSequence(sequence, list(range(i, sequence.length)))

            trajectory = Trajectory.read(results, name)

            overlaps = calculate_overlaps(trajectory.regions(), proxy.groundtruth(), proxy.size if self._burnin else None)

            grace = self._grace
            progress = len(proxy)

            for j, overlap in enumerate(overlaps):
                if overlap <= self._threshold and not proxy.groundtruth(j).is_empty():
                    grace = grace - 1
                    if grace == 0:
                        progress = j + 1 - self._grace  # subtract since we need actual point of the failure
                        break
                else:
                    grace = self._grace

            for j in range(progress):
                overlap = overlaps[j]

                tags = proxy.tags(j)
                if len(tags) == 0:
                    tags = ['empty']

                for t in tags:
                    accuracy_[t] += overlap
                    tags_count_[t] += 1

            robustness_ += progress
            total_ += len(proxy)

        seq_robustness = robustness_ / total_

        seq_accuracy = {}
        for t in accuracy_:
            seq_accuracy[t] = accuracy_[t] / tags_count_[t]

        # calculate weights for each attribute
        attribute_counter = Counter()
        for frame_idx in range(len(sequence)):
            tags = sequence.tags(frame_idx)
            if len(tags) == 0:
                tags = ['empty']
            for t in tags:
                attribute_counter[t] += 1

        return seq_accuracy, seq_robustness, attribute_counter

class AttributeDifficultyLevelMultiStart(SeparatableAnalysis):
    def __init__(self, fail_interval: int, burnin: int = 10, grace: int = 10, bounded: bool = True):
        super().__init__()
        self._burnin = burnin
        self._grace = grace
        self._bounded = bounded
        self._threshold = 0.1
        self._fail_interval = int(fail_interval)

    @property
    def name(self):
        return "Attribute difficulty level analysis"

    def parameters(self) -> Dict[str, Any]:
        return dict(burnin=self._burnin, grace=self._grace, bounded=self._bounded)

    def describe(self):
        return MeasureDescription("Attr. Difficulty Level", 0, 1, MeasureDescription.DESCENDING), \
            None

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiStartExperiment)

    def join(self, results: List[tuple]):
        attribute_difficulty = Counter()
        attribute_counter = Counter()
        for seq_tags_not_failed, seq_tags_count, seq_attr_count in results:
            
            for t in seq_tags_count:

                if t in seq_tags_not_failed:
                    seq_attr_difficulty = seq_tags_not_failed[t] / seq_tags_count[t]
                else:
                    seq_attr_difficulty = 0

                attribute_difficulty[t] += seq_attr_difficulty * seq_attr_count[t]
                attribute_counter[t] += seq_attr_count[t]

        for t in attribute_difficulty:
            attribute_difficulty[t] /= attribute_counter[t]

        return attribute_difficulty, attribute_counter


    def compute_partial(self, tracker: Tracker, experiment: Experiment, sequence: Sequence):

        results = experiment.results(tracker, sequence)

        forward, backward = find_anchors(sequence, experiment.anchor)

        if len(forward) == 0 and len(backward) == 0:
            raise RuntimeError("Sequence does not contain any anchors")

        tags_count = Counter()
        tags_not_failed = Counter()
        for i, reverse in [(f, False) for f in forward] + [(f, True) for f in backward]:
            name = "%s_%08d" % (sequence.name, i)

            if not Trajectory.exists(results, name):
                raise MissingResultsException()

            if reverse:
                proxy = FrameMapSequence(sequence, list(reversed(range(0, i + 1))))
            else:
                proxy = FrameMapSequence(sequence, list(range(i, sequence.length)))

            trajectory = Trajectory.read(results, name)

            overlaps = calculate_overlaps(trajectory.regions(), proxy.groundtruth(), proxy.size if self._burnin else None)

            grace = self._grace
            progress = len(proxy)

            for j, overlap in enumerate(overlaps):
                if overlap <= self._threshold and not proxy.groundtruth(j).is_empty():
                    grace = grace - 1
                    if grace == 0:
                        progress = j + 1 - self._grace  # subtract since we need actual point of the failure
                        break
                else:
                    grace = self._grace
            
            for j in range(progress):
                tags = proxy.tags(j)
                if len(tags) == 0:
                    tags = ['empty']

                for t in tags:
                    tags_count[t] += 1
                    if progress == len(proxy) or j < progress - self._fail_interval:
                        tags_not_failed[t] += 1

        attribute_counter = Counter()
        for frame_idx in range(len(sequence)):
            tags = sequence.tags(frame_idx)
            if len(tags) == 0:
                tags = ['empty']
            for t in tags:
                attribute_counter[t] += 1

        return tags_not_failed, tags_count, attribute_counter
