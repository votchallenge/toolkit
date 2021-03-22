import math
from typing import List, Tuple, Any

import numpy as np

from attributee import Integer, Boolean, Float, Include

from vot.tracker import Tracker, Trajectory
from vot.dataset import Sequence
from vot.dataset.proxy import FrameMapSequence
from vot.experiment import Experiment
from vot.experiment.multistart import MultiStartExperiment, find_anchors
from vot.region import calculate_overlaps
from vot.analysis import MissingResultsException, Measure, Plot, Analysis, Axes, \
    Sorting, SeparableAnalysis, Curve, Point, analysis_registry, SequenceAggregator
from vot.utilities.data import Grid

def compute_eao_partial(overlaps: List, success: List[bool], curve_length: int):
    phi = curve_length * [float(0)]
    active = curve_length * [float(0)]

    for o, success in zip(overlaps, success):

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


@analysis_registry.register("multistart_ar")
class AccuracyRobustness(SeparableAnalysis):

    burnin = Integer(default=10, val_min=0)
    grace = Integer(default=10, val_min=0)
    bounded = Boolean(default=True)
    threshold = Float(default=0.1, val_min=0, val_max=1)

    @property
    def title(self):
        return "AR Analysis"

    def describe(self):
        return Measure("Accuracy", "A", minimal=0, maximal=1, direction=Sorting.DESCENDING), \
             Measure("Robustness", "R", minimal=0, direction=Sorting.DESCENDING), \
             Point("AR plot", dimensions=2, abbreviation="AR",
                minimal=(0, 0), maximal=(1, 1), labels=("Robustness", "Accuracy"), trait="ar"), \
             None, None

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiStartExperiment)

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:

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

            overlaps = calculate_overlaps(trajectory.regions(), proxy.groundtruth(), (proxy.size) if self.burnin else None)

            grace = self.grace
            progress = len(proxy)

            for j, overlap in enumerate(overlaps):
                if overlap <= self.threshold and not proxy.groundtruth(j).is_empty():
                    grace = grace - 1
                    if grace == 0:
                        progress = j + 1 - self.grace  # subtract since we need actual point of the failure
                        break
                else:
                    grace = self.grace

            robustness += progress  # simplified original equation: len(proxy) * (progress / len(proxy))
            accuracy += sum(overlaps[0:progress])
            total += len(proxy)

        ar = (robustness / total, accuracy / robustness if robustness > 0 else 0)

        return accuracy / robustness if robustness > 0 else 0, robustness / total, ar, robustness, len(sequence)

@analysis_registry.register("multistart_average_ar")
class AverageAccuracyRobustness(SequenceAggregator):

    analysis = Include(AccuracyRobustness)

    @property
    def title(self):
        return "AR Analysis"

    def dependencies(self):
        return self.analysis, 

    def describe(self):
        return Measure("Accuracy", "A", minimal=0, maximal=1, direction=Sorting.DESCENDING), \
             Measure("Robustness", "R", minimal=0, direction=Sorting.DESCENDING), \
             Point("AR plot", dimensions=2, abbreviation="AR",
                minimal=(0, 0), maximal=(1, 1), labels=("Robustness", "Accuracy"), trait="ar"), \
             None, None

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiStartExperiment)

    def aggregate(self, tracker: Tracker, sequences: List[Sequence], results: Grid):
        total_accuracy = 0
        total_robustness = 0
        weight_accuracy = 0
        weight_robustness = 0

        for accuracy, robustness, _, accuracy_w, robustness_w in results:
            total_accuracy += accuracy * accuracy_w
            total_robustness += robustness * robustness_w
            weight_accuracy += accuracy_w
            weight_robustness += robustness_w

        ar = (total_robustness / weight_robustness, total_accuracy / weight_accuracy)

        return total_accuracy / weight_accuracy, total_robustness / weight_robustness, ar, weight_accuracy, weight_robustness

@analysis_registry.register("multistart_fragments")
class MultiStartFragments(SeparableAnalysis):

    burnin = Integer(default=10, val_min=0)
    grace = Integer(default=10, val_min=0)
    bounded = Boolean(default=True)
    threshold = Float(default=0.1, val_min=0, val_max=1)

    @property
    def title(self):
        return "Fragment Analysis"

    def describe(self):
        return Curve("Success", 2, "Sc", minimal=(0, 0), maximal=(1,1), trait="points"), Curve("Accuracy", 2, "Ac", minimal=(0, 0), maximal=(1,1), trait="points")

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiStartExperiment)

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:

        results = experiment.results(tracker, sequence)

        forward, backward = find_anchors(sequence, experiment.anchor)

        if not forward and not backward:
            raise RuntimeError("Sequence does not contain any anchors")

        accuracy = []
        success = []

        for i, reverse in [(f, False) for f in forward] + [(f, True) for f in backward]:
            name = "%s_%08d" % (sequence.name, i)

            if not Trajectory.exists(results, name):
                raise MissingResultsException()

            if reverse:
                proxy = FrameMapSequence(sequence, list(reversed(range(0, i + 1))))
            else:
                proxy = FrameMapSequence(sequence, list(range(i, sequence.length)))

            trajectory = Trajectory.read(results, name)

            overlaps = calculate_overlaps(trajectory.regions(), proxy.groundtruth(), (proxy.size) if self.burnin else None)

            grace = self.grace
            progress = len(proxy)

            for j, overlap in enumerate(overlaps):
                if overlap <= self.threshold and not proxy.groundtruth(j).is_empty():
                    grace = grace - 1
                    if grace == 0:
                        progress = j + 1 - self.grace  # subtract since we need actual point of the failure
                        break
                else:
                    grace = self.grace

            success.append( (i / len(sequence), progress / len(proxy)))
            accuracy.append( (i / len(sequence), sum(overlaps[0:progress] / len(proxy))))

        return success, accuracy

# TODO: remove high
@analysis_registry.register("multistart_eao_curves")
class EAOCurves(SeparableAnalysis):

    burnin = Integer(default=10, val_min=0)
    grace = Integer(default=10, val_min=0)
    bounded = Boolean(default=True)
    threshold = Float(default=0.1, val_min=0, val_max=1)

    high = Integer()

    @property
    def title(self):
        return "EAO Curve"

    def describe(self):
        return Plot("Expected average overlap", "EAO", minimal=0, maximal=1, wrt="frames", trait="eao"),

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiStartExperiment)

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:

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

            overlaps = calculate_overlaps(trajectory.regions(), proxy.groundtruth(), proxy.size if self.burnin else None)

            grace = self.grace
            progress = len(proxy)

            for j, overlap in enumerate(overlaps):
                if overlap <= self.threshold and not proxy.groundtruth(j).is_empty():
                    grace = grace - 1
                    if grace == 0:
                        progress = j + 1 - self.grace  # subtract since we need actual point of the failure
                        break
                else:
                    grace = self.grace

            success = True
            if progress < len(overlaps):
                # tracker has failed during this run
                overlaps[progress:] = (len(overlaps) - progress) * [float(0)]
                success = False

            overlaps_all.append(overlaps)
            success_all.append(success)

        return compute_eao_partial(overlaps_all, success_all, self.high), 1

#TODO: remove high
@analysis_registry.register("multistart_eao_curve")
class EAOCurve(SequenceAggregator):

    curves = Include(EAOCurves)
    
    @property
    def title(self):
        return "EAO Curve"

    def describe(self):
        return Plot("Expected average overlap", "EAO", minimal=0, maximal=1, wrt="frames", trait="eao"),

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiStartExperiment)

    def dependencies(self):
        return self.curves,

    def aggregate(self, tracker: Tracker, sequences: List[Sequence], results: Grid) -> Tuple[Any]:

        eao_curve = self.curves.high * [float(0)]
        eao_weights = self.curves.high * [float(0)]

        for (seq_eao_curve, eao_active), seq_w in results:
            for i, (eao_, active_) in enumerate(zip(seq_eao_curve, eao_active)):
                eao_curve[i] += eao_ * active_ * seq_w
                eao_weights[i] += active_ * seq_w

        return [eao_ / w_ if w_ > 0 else 0 for eao_, w_ in zip(eao_curve, eao_weights)],

@analysis_registry.register("multistart_eao_score")
class EAOScore(Analysis):

    low = Integer()
    high = Integer()
    eaocurve = Include(EAOCurve)

    @property
    def title(self):
        return "EAO analysis"

    def describe(self):
        return Measure("Expected average overlap", "EAO", minimal=0, maximal=1, direction=Sorting.DESCENDING),

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiStartExperiment)

    def dependencies(self):
        return self.eaocurve,

    def compute(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence], dependencies: List[Grid]) -> Grid:

        return dependencies[0].foreach(lambda x, i, j: (float(np.mean(x[0][self.low:self.high + 1])), ) )

    @property
    def axes(self):
        return Axes.TRACKERS

