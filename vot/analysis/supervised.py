""" This module constans common analysis routines for supervised experiment, e.g. Accuracy-Robustness and EAO as
 defined in VOT papers.
"""

import math
from typing import List, Tuple, Any

import numpy as np

from attributee import Integer, Boolean, Float, Include

from vot.tracker import Tracker, Trajectory
from vot.dataset import Sequence
from vot.experiment import Experiment
from vot.experiment.multirun import SupervisedExperiment
from vot.region import Region, calculate_overlaps
from vot.analysis import MissingResultsException, Measure, Point, is_special, Plot, Analysis, \
    Sorting, SeparableAnalysis, SequenceAggregator, TrackerSeparableAnalysis, Axes
from vot.utilities.data import Grid

def compute_accuracy(trajectory: List[Region], sequence: Sequence, burnin: int = 10, 
    ignore_unknown: bool = True, bounded: bool = True) -> float:
    """ Computes accuracy of a tracker on a given sequence. Accuracy is defined as mean overlap of the tracker
    region with the groundtruth region. The overlap is computed only for frames where the tracker is not in
    initialization or failure state. The overlap is computed only for frames after the burnin period.

    Args:
        trajectory (List[Region]): Tracker trajectory.
        sequence (Sequence): Sequence to compute accuracy on.
        burnin (int, optional): Burnin period. Defaults to 10.
        ignore_unknown (bool, optional): Ignore unknown regions. Defaults to True.
        bounded (bool, optional): Consider only first N frames. Defaults to True.

    Returns:
        float: Accuracy.
    """

    overlaps = np.array(calculate_overlaps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    mask = np.ones(len(overlaps), dtype=bool)

    for i, region in enumerate(trajectory):
        if is_special(region, Trajectory.UNKNOWN) and ignore_unknown:
            mask[i] = False
        elif is_special(region, Trajectory.INITIALIZATION):
            for j in range(i, min(len(trajectory), i + burnin)):
                mask[j] = False
        elif is_special(region, Trajectory.FAILURE):
            mask[i] = False
    
    if any(mask):
        return np.mean(overlaps[mask]), np.sum(mask)
    else:
        return 0, 0

def count_failures(trajectory: List[Region]) -> Tuple[int, int]:
    """Counts number of failures in a trajectory. Failure is defined as a frame where the tracker is in failure state."""
    return len([region for region in trajectory if is_special(region, Trajectory.FAILURE)]), len(trajectory)


def locate_failures_inits(trajectory: List[Region]) -> Tuple[int, int]:
    """Locates failures and initializations in a trajectory. Failure is defined as a frame where the tracker is in failure state."""
    return [i for i, region in enumerate(trajectory) if is_special(region, Trajectory.FAILURE)], \
            [i for i, region in enumerate(trajectory) if is_special(region, Trajectory.INITIALIZATION)]

def compute_eao_curve(overlaps: List, weights: List[float], success: List[bool]):
    """Computes EAO curve from a list of overlaps, weights and success flags."""
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
    
class AccuracyRobustness(SeparableAnalysis):
    """Accuracy-Robustness analysis. Computes accuracy and robustness of a tracker on a given sequence. 
    Accuracy is defined as mean overlap of the tracker region with the groundtruth region. The overlap is computed only for frames where the tracker is not in
    initialization or failure state. The overlap is computed only for frames after the burnin period.
    Robustness is defined as a number of failures divided by the total number of frames.
    """

    sensitivity = Float(default=30, val_min=1)
    burnin = Integer(default=10, val_min=0)
    ignore_unknown = Boolean(default=True)
    bounded = Boolean(default=True)

    @property
    def _title_default(self):
        """Returns title of the analysis."""
        return "AR analysis"

    def describe(self):
        """Returns description of the analysis."""
        return Measure("Accuracy", "A", minimal=0, maximal=1, direction=Sorting.DESCENDING), \
             Measure("Robustness", "R", minimal=0, direction=Sorting.ASCENDING), \
             Point("AR plot", dimensions=2, abbreviation="AR", minimal=(0, 0), \
                maximal=(1, 1), labels=("Robustness", "Accuracy"), trait="ar"), \
             None

    def compatible(self, experiment: Experiment):
        """Returns True if the analysis is compatible with the experiment. Only SupervisedExperiment is compatible."""
        return isinstance(experiment, SupervisedExperiment)

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:
        """Computes accuracy and robustness of a tracker on a given sequence. 
        
        Args:
            experiment (Experiment): Experiment.
            tracker (Tracker): Tracker.
            sequence (Sequence): Sequence.
            dependencies (List[Grid]): Dependencies.
            
        Returns:
            Tuple[Any]: Accuracy, robustness, AR, number of frames.
        """
        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        accuracy = 0
        failures = 0
        for trajectory in trajectories:
            failures += count_failures(trajectory.regions())[0]
            accuracy += compute_accuracy(trajectory.regions(), sequence, self.burnin, self.ignore_unknown, self.bounded)[0]

        failures /= len(trajectories)
        accuracy /= len(trajectories)

        ar = (math.exp(- (float(failures) / len(sequence)) * self.sensitivity), accuracy)

        return accuracy, failures, ar, len(sequence)

class AverageAccuracyRobustness(SequenceAggregator):
    """Average accuracy-robustness analysis. Computes average accuracy and robustness of a tracker on a given sequence. 

    Accuracy is defined as mean overlap of the tracker region with the groundtruth region. The overlap is computed only for frames where the tracker is not in
    initialization or failure state. The overlap is computed only for frames after the burnin period.
    Robustness is defined as a number of failures divided by the total number of frames. 
    The analysis is computed as an average of accuracy and robustness over all sequences.
    """

    analysis = Include(AccuracyRobustness)

    @property
    def _title_default(self):
        """Returns title of the analysis."""
        return "AR Analysis"

    def dependencies(self):
        """Returns dependencies of the analysis."""
        return self.analysis,

    def describe(self):
        """Returns description of the analysis."""
        return Measure("Accuracy", "A", minimal=0, maximal=1, direction=Sorting.DESCENDING), \
             Measure("Robustness", "R", minimal=0, direction=Sorting.ASCENDING), \
             Point("AR plot", dimensions=2, abbreviation="AR", minimal=(0, 0), \
                maximal=(1, 1), labels=("Robustness", "Accuracy"), trait="ar"), \
             None

    def compatible(self, experiment: Experiment):
        """Returns True if the analysis is compatible with the experiment. Only SupervisedExperiment is compatible."""
        return isinstance(experiment, SupervisedExperiment)

    def aggregate(self, tracker: Tracker, sequences: List[Sequence], results: Grid):
        """Aggregates results of the analysis.
        
        Args:
            tracker (Tracker): Tracker.
            sequences (List[Sequence]): List of sequences.
            results (Grid): Results of the analysis.
            
        Returns:
            Tuple[Any]: Accuracy, robustness, AR, number of frames.
        """

        failures = 0
        accuracy = 0
        weight_total = 0

        for a, f, _, w in results:
            failures += f * w
            accuracy += a * w
            weight_total += w

        failures /= weight_total
        accuracy /= weight_total
        length = weight_total / len(results)

        ar = (math.exp(- (failures / length) * self.analysis.sensitivity), accuracy)

        return accuracy, failures, ar, length

class EAOCurve(TrackerSeparableAnalysis):
    """Expected Average Overlap curve analysis. Computes expected average overlap of a tracker on a given sequence.
    The overlap is computed only for frames where the tracker is not in initialization or failure state.
    The overlap is computed only for frames after the burnin period.
    The analysis is computed as an average of accuracy and robustness over all sequences.
    """

    burnin = Integer(default=10, val_min=0)
    bounded = Boolean(default=True)

    @property
    def _title_default(self):
        """Returns title of the analysis."""
        return "EAO Curve"

    def describe(self):
        """Returns description of the analysis."""
        return Plot("Expected Average Overlap", "EAO", minimal=0, maximal=1, trait="eao"),

    def compatible(self, experiment: Experiment):
        """Returns True if the analysis is compatible with the experiment. Only SupervisedExperiment is compatible."""
        return isinstance(experiment, SupervisedExperiment)

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequences: List[Sequence], dependencies: List[Grid]) -> Tuple[Any]:
        """Computes expected average overlap of a tracker on a given sequence.
        
        Args:
            experiment (Experiment): Experiment.
            tracker (Tracker): Tracker.
            sequences (List[Sequence]): List of sequences.
            dependencies (List[Grid]): Dependencies.
            
        Returns:
            Tuple[Any]: Expected average overlap.
        """

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

        return compute_eao_curve(overlaps_all, weights_all, success_all),

class EAOScore(Analysis):
    """Expected Average Overlap score analysis. The analysis is computed as an average of EAO scores over multiple sequences.
    """

    eaocurve = Include(EAOCurve)
    low = Integer()
    high = Integer()

    @property
    def _title_default(self):
        """Returns title of the analysis."""
        return "EAO analysis"

    def describe(self):
        """Returns description of the analysis."""
        return Measure("Expected average overlap", "EAO", 0, 1, Sorting.DESCENDING),

    def compatible(self, experiment: Experiment):
        """Returns True if the analysis is compatible with the experiment. Only SupervisedExperiment is compatible."""
        return isinstance(experiment, SupervisedExperiment)

    def dependencies(self):
        """Returns dependencies of the analysis."""
        return self.eaocurve,

    def compute(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence], dependencies: List[Grid]) -> Grid:
        """Computes expected average overlap of a tracker on a given sequence.

        Args:
            experiment (Experiment): Experiment.
            trackers (List[Tracker]): List of trackers.
            sequences (List[Sequence]): List of sequences.
            dependencies (List[Grid]): Dependencies.

        Returns:
            Grid: Expected average overlap.
        """
        return dependencies[0].foreach(lambda x, i, j: (float(np.mean(x[0][self.low:self.high + 1])), ) )

    @property
    def axes(self):
        """Returns axes of the analysis."""
        return Axes.TRACKERS