from typing import List

from vot.tracker import Tracker, Trajectory
from vot.dataset import Sequence
from vot.dataset.proxy import FrameMapSequence
from vot.experiment import Experiment
from vot.experiment.multirun import MultiRunExperiment, SupervisedExperiment
from vot.experiment.multistart import MultiStartExperiment, find_anchors
from vot.analysis import SeparatablePerformanceMeasure, MissingResultsException, MeasureDescription
from vot.analysis.routines import count_failures, compute_accuracy

class AverageAccuracy(SeparatablePerformanceMeasure):

    def __init__(self, burnin: int = 10, ignore_unknown: bool = True, bounded: bool = True):
        self._burnin = burnin
        self._ignore_unknown = ignore_unknown
        self._bounded = bounded

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    # TODO: turn off weighted average
    def join(self, results: List[tuple]):
        accuracy = 0
        frames = 0

        for a, n in results:
            accuracy = accuracy + a
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

class FailureCount(SeparatablePerformanceMeasure):

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, SupervisedExperiment)

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
            failures = failures + count_failures(trajectory.regions())

        return failures / len(trajectories), len(trajectories[0])

class AccuracyRobustness(SeparatablePerformanceMeasure):

    def __init__(self, sensitivity: int = 30, burnin: int = 10, ignore_unknown: bool = True, bounded: bool = True):
        self._sensitivity = sensitivity
        self._burnin = burnin
        self._ignore_unknown = ignore_unknown
        self._bounded = bounded

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, SupervisedExperiment)

    def join(self, results: List[tuple]):
        failures = 0
        accuracy = 0
        weight_total = 0

        for a, f, w in results:
            failures += f * w
            accuracy += a * w
            weight_total += w

        return accuracy / weight_total, failures / weight_total, weight_total

    def compute_partial(self, tracker: Tracker, experiment: Experiment, sequence: Sequence):
        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        accuracy = 0
        failures = 0
        for trajectory in trajectories:
            failures += count_failures(trajectory.regions())[0]
            accuracy += compute_accuracy(trajectory.regions(), sequence, self._burnin, self._ignore_unknown, self._bounded)[0]

        return accuracy / len(trajectories), failures / len(trajectories), len(trajectories[0])

class AccuracyRobustnessMultiStart(SeparatablePerformanceMeasure):

    def __init__(self, burnin: int = 10, grace: int = 10, bounded: bool = True):
        self._burnin = burnin
        self._grace = grace
        self._bounded = bounded
        self._threshold = 0.1

    @classmethod
    def describe(cls):
        return MeasureDescription("Accuracy", 0, 1, MeasureDescription.DESCENDING), \
            MeasureDescription("Robustness", 0, 1, MeasureDescription.DESCENDING), \
            None

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiStartExperiment)

    def join(self, results: List[tuple]):
        total_accuracy = 0
        total_robustness = 0
        total = 0

        for accuracy, robustness, weight in results:
            total_accuracy += accuracy * weight
            total_robustness += robustness * weight
            total += weight

        return total_accuracy / total, total_robustness / total, total

    def compute_partial(self, tracker: Tracker, experiment: Experiment, sequence: Sequence):

        from vot.region.utils import calculate_overlaps

        results = experiment.results(tracker, sequence)

        forward, backward = find_anchors(sequence, experiment.anchor)

        if len(forward) == 0 and len(backward) == 0:
            raise RuntimeError("Sequence does not contain any anchors")

        robustness = 0
        accuracy = 0        
        total = 0
        for i, reverse in [(f, False) for f in forward] + [(f, True) for f in backward]:
            name = "%s_%08d" % (sequence.name, i)

            if not Trajectory.exists(results, name):
                raise MissingResultsException()

            if reverse:
                proxy = FrameMapSequence(sequence, list(reversed(range(0, i))))
            else:
                proxy = FrameMapSequence(sequence, list(range(i, sequence.length)))

            trajectory = Trajectory.read(results, name)

            overlaps = calculate_overlaps(trajectory.regions(), proxy.groundtruth(), proxy.size if self._burnin else None)

            grace = self._grace
            progress = len(proxy)

            for j, overlap in enumerate(overlaps):
                if overlap <= self._threshold:
                    grace = grace - 1
                    if grace == 0:
                        progress = j - self._grace  # subtract since we need actual point of the failure
                        break
                else:
                    grace = self._grace

            robustness += progress  # simplified original equation: len(proxy) * (progress / len(proxy))
            accuracy += len(proxy) * (sum(overlaps[0:progress]) / progress)
            total += len(proxy)
            
        return accuracy / total, robustness / total, len(sequence)