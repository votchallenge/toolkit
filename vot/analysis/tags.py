
import numpy as np
import typing
from typing import List, Tuple
from collections import Counter

from vot.tracker import Tracker, Trajectory
from vot.dataset import Sequence
from vot.dataset.proxy import FrameMapSequence
from vot.region import Region, Special, calculate_overlaps
from vot.experiment import Experiment
from vot.experiment.multirun import SupervisedExperiment
from vot.experiment.multistart import MultiStartExperiment, find_anchors
from vot.analysis import MissingResultsException, \
    Plot, Point, is_special, public, Axis, Sorting, Measure, SequenceAveragingAnalysis
from vot.utilities.attributes import String, Float, Integer, Boolean, List


class AttributeMultiStart(SequenceAveragingAnalysis):

    burnin = Integer(default=10, val_min=0)
    grace = Integer(default=10, val_min=0)
    bounded = Boolean(default=True)
    threshold = Float(default=0.1, val_min=0, val_max=1)

    tags = List(String())

    @property
    def name(self):
        return "AR per-attribute analysis"

    def describe(self):
        return [Measure("Accuracy", "A", minimal=0, maximal=1, direction=Sorting.DESCENDING) for t in self.tags], \
            [Measure("Robustness", "R", minimal=0, maximal=1, direction=Sorting.DESCENDING) for t in self.tags], \
            None

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiStartExperiment)

    def collapse(self, tracker: Tracker, sequences: typing.List[Sequence], results: typing.List[tuple]):
        accuracy = Counter()
        robustness = Counter()
        attribute_total = Counter()

        for seq_acc, seq_rob, seq_attr_count in results:
            for t in seq_attr_count:
                accuracy[t] += (seq_acc[t] if t in seq_acc else 0) * seq_attr_count[t]
                robustness[t] += seq_rob * seq_attr_count[t]
                attribute_total[t] += seq_attr_count[t]

        return [accuracy[t] / attribute_total[t] for t in self.tags], \
            [robustness[t] / attribute_total[t] for t in self.tags], \
            [attribute_total[t] for t in self.tags]

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence):

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

class AttributeDifficultyLevelMultiStart(SequenceAveragingAnalysis):

    burnin = Integer(default=10, val_min=0)
    grace = Integer(default=10, val_min=0)
    bounded = Boolean(default=True)
    threshold = Float(default=0.1, val_min=0, val_max=1)

    fail_interval = Integer()
    tags = List(String())

    @property
    def name(self):
        return "Attribute difficulty"

    def describe(self):
        return [Measure("Difficulty", "D", minimal=0, maximal=1, direction=Sorting.DESCENDING) for t in self.tags], \
            None

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiStartExperiment)

    def collapse(self, tracker: Tracker, sequences: typing.List[Sequence], results: typing.List[tuple]):
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

        return [attribute_difficulty[t] / attribute_counter[t] for t in self.tags], [attribute_counter[t] for t in self.tags]


    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence):

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
            
            for j in range(progress):
                tags = proxy.tags(j)
                if len(tags) == 0:
                    tags = ['empty']

                for t in tags:
                    tags_count[t] += 1
                    if progress == len(proxy) or j < progress - self.fail_interval:
                        tags_not_failed[t] += 1

        attribute_counter = Counter()
        for frame_idx in range(len(sequence)):
            tags = sequence.tags(frame_idx)
            if len(tags) == 0:
                tags = ['empty']
            for t in tags:
                attribute_counter[t] += 1

        return tags_not_failed, tags_count, attribute_counter
