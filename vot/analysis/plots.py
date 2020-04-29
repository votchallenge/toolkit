

import numpy as np
from typing import List, Dict, Any, Tuple

from vot.tracker import Tracker, Trajectory
from vot.dataset import Sequence
from vot.dataset.proxy import FrameMapSequence
from vot.region import Region, Special
from vot.experiment import Experiment
from vot.experiment.multirun import SupervisedExperiment
from vot.experiment.multistart import MultiStartExperiment, find_anchors
from vot.analysis import Analysis, MissingResultsException, Plot, is_special
from vot.utilities import to_number, to_logical

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
    
class EAOCurve(Analysis):
    def __init__(self, burnin: int = 10, grace: int = 10, bounded: bool = True):
        super().__init__()
        self._burnin = to_number(burnin, min_n=0)
        self._grace = to_number(grace, min_n=0)
        self._bounded = to_logical(bounded)

    @property
    def name(self):
        return "EAO Curve"

    def parameters(self) -> Dict[str, Any]:
        return dict(burnin=self._burnin, grace=self._grace, bounded=self._bounded)

    def describe(self):
        return Plot("Expected Average Overlap", "EAO", minimal=0, maximal=1),

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, SupervisedExperiment)

    def compute(self, tracker: Tracker, experiment: Experiment, sequences: List[Sequence]):
        from vot.region.utils import calculate_overlaps

        overlaps_all = []
        weights_all = []
        success_all = []

        for sequence in sequences:

            trajectories = experiment.gather(tracker, sequence)

            if len(trajectories) == 0:
                raise MissingResultsException()

            for trajectory in trajectories:

                overlaps = calculate_overlaps(trajectory.regions(), sequence.groundtruth(), (sequence.size) if self._bounded else None)
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

        return compute_eao_curve(overlaps_all, weights_all, success_all),

class EAOCurveMultiStart(Analysis):

    def __init__(self, burnin: int = 10, grace: int = 10, bounded: bool = True, threshold: float = 0.1):
        super().__init__()
        self._burnin = to_number(burnin, min_n=0)
        self._grace = to_number(grace, min_n=0)
        self._bounded = to_logical(bounded)
        self._threshold = to_number(threshold, min_n=0, max_n=1, conversion=float)
        
    @property
    def name(self):
        return "EAO Curve"

    def parameters(self) -> Dict[str, Any]:
        return dict(burnin=self._burnin, grace=self._grace, bounded=self._bounded, threshold=self._threshold)

    def describe(self):
        return Plot("Expected Average Overlap", "EAO", minimal=0, maximal=1),

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiStartExperiment)

    def compute(self, tracker: Tracker, experiment: Experiment, sequences: List[Sequence]):

        from vot.region.utils import calculate_overlaps

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
                weights_per_run.append(len(proxy))

            for w in weights_per_run:
                weights_all.append((w / sum(weights_per_run)) * len(sequence))

            frames_total += len(sequence)

        weights_all = [w / frames_total for w in weights_all]

        return compute_eao_curve(overlaps_all, weights_all, success_all), 