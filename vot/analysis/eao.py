
import numpy as np
from typing import List, Tuple

from vot.tracker import Tracker, Trajectory
from vot.dataset import Sequence
from vot.dataset.proxy import FrameMapSequence
from vot.region import Region, Special, calculate_overlaps
from vot.experiment import Experiment
from vot.experiment.multirun import SupervisedExperiment
from vot.experiment.multistart import MultiStartExperiment, find_anchors
from vot.analysis import TrackerSeparableAnalysis, DependentAnalysis, MissingResultsException, \
    Plot, Point, is_special, Axis, Sorting, Measure, SequenceAveragingAnalysis
from vot.utilities import alias
from vot.utilities.data import Grid
from vot.utilities.attributes import Float, Integer, Boolean, Include

def locate_failures_inits(trajectory: List[Region]) -> Tuple[int, int]:
    return [i for i, region in enumerate(trajectory) if is_special(region, Special.FAILURE)], \
            [i for i, region in enumerate(trajectory) if is_special(region, Special.INITIALIZATION)]

def compute_eao_curve(overlaps: List, weights: List[float], success: List[bool]):
    max_length = max([len(el) for el in overlaps])
    total_runs = len(overlaps)
    
    overlaps_array = np.zeros((total_runs, max_length), dtype=np.float32)
    mask_array = np.zeros((total_runs, max_length), dtype=np.float32)  # mask out frames which are not considered in EAO calculation
    weights_vector = np.reshape(np.array(weights, dtype=np.float32), (len(weights), 1))  # weight of each run

    for i, (o, success) in enumerate(zip(overlaps, success)):
        overlaps_array[i, :len(o)] = np.array(o)
        if not success:
            # tracker has failed during this run - fill zeros until the end of the run
            mask_array[i, :] = 1
        else:
            # tracker has successfully tracked to the end - consider only this part of the sequence
            mask_array[i, :len(o)] = 1

    overlaps_array_sum = overlaps_array.copy()
    for j in range(1, overlaps_array_sum.shape[1]):
        overlaps_array_sum[:, j] = np.mean(overlaps_array[:, 1:j+1], axis=1)
    
    return np.sum(weights_vector * overlaps_array_sum * mask_array, axis=0) / np.sum(mask_array * weights_vector, axis=0).tolist()
    
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

@alias("EAO curve", "EAOCurve", "eaocurve", "EAOCurveSupervised")
class EAOCurve(TrackerSeparableAnalysis):

    burnin = Integer(default=10, val_min=0)
    bounded = Boolean(default=True)

    @property
    def title(self):
        return "EAO Curve"

    def describe(self):
        return Plot("Expected Average Overlap", "EAO", minimal=0, maximal=1, trait="eao"),

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, SupervisedExperiment)

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequences: List[Sequence]):

        overlaps_all = []
        weights_all = []
        success_all = []

        for sequence in sequences:

            trajectories = experiment.gather(tracker, sequence)

            if len(trajectories) == 0:
                raise MissingResultsException()

            for trajectory in trajectories:

                overlaps = calculate_overlaps(trajectory.regions(), sequence.groundtruth(), (sequence.size) if self.bounded else None)
                fail_idxs, init_idxs = locate_failures_inits(trajectory.regions())

                if len(fail_idxs) > 0:

                    for i in range(len(fail_idxs)):
                        overlaps_all.append(overlaps[init_idxs[i]:fail_idxs[i]])
                        success_all.append(False)
                        weights_all.append(1)

                    # handle last initialization
                    if len(init_idxs) > len(fail_idxs):
                        # tracker was initilized, but it has not failed until the end of the sequence
                        overlaps_all.append(overlaps[init_idxs[-1]:])
                        success_all.append(True)
                        weights_all.append(1)

                else:
                    overlaps_all.append(overlaps)
                    success_all.append(True)
                    weights_all.append(1)

        result = Grid((1,1))
        result[0, 0] = (compute_eao_curve(overlaps_all, weights_all, success_all),)

        return result

class EAOCurveMultiStart2(TrackerSeparableAnalysis):

    burnin = Integer(default=10, val_min=0)
    grace = Integer(default=10, val_min=0)
    bounded = Boolean(default=True)
    threshold = Float(default=0.1, val_min=0, val_max=1)

    @property
    def title(self):
        return "EAO Curve"

    def describe(self):
        return Plot("Expected Average Overlap", "EAO", minimal=0, maximal=1),

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiStartExperiment)

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequences: List[Sequence]):

        overlaps_all = []
        weights_all = []
        success_all = []
        frames_total = 0

        for sequence in sequences:

            results = experiment.results(tracker, sequence)

            forward, backward = find_anchors(sequence, experiment.anchor)

            if len(forward) == 0 and len(backward) == 0:
                raise RuntimeError("Sequence does not contain any anchors")

            weights_per_run = []
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
                weights_per_run.append(len(proxy))

            for w in weights_per_run:
                weights_all.append((w / sum(weights_per_run)) * len(sequence))

            frames_total += len(sequence)

        weights_all = [w / frames_total for w in weights_all]

        return compute_eao_curve(overlaps_all, weights_all, success_all),

@alias("EAO score", "EAOScore", "eaoscore", "EAOScoreSupervised")
class EAOScore(DependentAnalysis):

    eaocurve = Include(EAOCurve)
    low = Integer()
    high = Integer()

    @property
    def title(self):
        return "EAO analysis"

    def describe(self):
        return Measure("Expected average overlap", "EAO", 0, 1, Sorting.DESCENDING),

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, SupervisedExperiment)

    def dependencies(self):
        return self.eaocurve,

    def join(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence], results: List[Grid]):
        return [(float(np.mean(x[0][self.low:self.high + 1])), ) for x in results[0]]

    def axes(self):
        return Axis.TRACKERS,

# TODO: remove low, high
@alias("EAO curve (multi-start)", "eao_curve_multistart", "EAOCurveMultiStart")
class EAOCurveMultiStart(SequenceAveragingAnalysis):

    burnin = Integer(default=10, val_min=0)
    grace = Integer(default=10, val_min=0)
    bounded = Boolean(default=True)
    threshold = Float(default=0.1, val_min=0, val_max=1)

    low = Integer()
    high = Integer()

    @property
    def title(self):
        return "EAO Curve"

    def describe(self):
        return Plot("Expected average overlap", "EAO", minimal=0, maximal=1, wrt="frames", trait="eao"),

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiStartExperiment)

    def collapse(self, tracker: Tracker, sequences: List[Sequence], results: Grid):
        eao_curve = self.high * [float(0)]
        eao_weights = self.high * [float(0)]

        for (seq_eao_curve, eao_active), seq_w in results:
            for i, (eao_, active_) in enumerate(zip(seq_eao_curve, eao_active)):
                eao_curve[i] += eao_ * active_ * seq_w
                eao_weights[i] += active_ * seq_w

        return [eao_ / w_ if w_ > 0 else 0 for eao_, w_ in zip(eao_curve, eao_weights)],

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence):

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

@alias("EAO score (multi-start)", "eao_score_multistart", "EAOScoreMultiStart")
class EAOScoreMultiStart(DependentAnalysis):

    low = Integer()
    high = Integer()
    eaocurve = Include(EAOCurveMultiStart)

    @property
    def title(self):
        return "EAO analysis"

    def describe(self):
        return Measure("Expected average overlap", "EAO", minimal=0, maximal=1, direction=Sorting.DESCENDING),

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiStartExperiment)

    def dependencies(self):
        return self.eaocurve,

    def join(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence], results: List[Grid]):
        joined = Grid((len(trackers), ))

        for i, result in enumerate(results[0]):
            if result is None:
                continue
            joined[i] = (float(np.mean(result[0][self.low:self.high + 1])), )
        return joined

    def axes(self):
        return Axis.TRACKERS,

