
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
        return Measure("Accuracy", "A", 0, 1, Sorting.DESCENDING), \
            Measure("Robustness", "R", 0, 1, Sorting.DESCENDING), \
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
        return "Attribute difficulty"

    def describe(self):
        return Measure("Difficulty", "D", 0, 1, Sorting.DESCENDING), \
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

