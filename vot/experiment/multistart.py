"""This module implements the multistart experiment."""

from typing import Callable

from attributee import String

from vot.dataset import Sequence
from vot.dataset.proxy import FrameMapSequence
from vot.region import Special
from vot.experiment import Experiment
from vot.tracker import ObjectQuery, Tracker, Trajectory

def find_anchors(sequence: Sequence, anchor="anchor"):
    """Find anchor frames in the sequence. Anchor frames are frames where the given
    object is visible and can be used for initialization.

    :param sequence: The sequence to be scanned.
    :type sequence: Sequence
    :param anchor: The name of the object to be used as an anchor. Defaults to "anchor".
    :type anchor: str, optional

    :returns: A tuple containing two lists of frames. The first list contains forward anchors, the second list contains backward anchors.
    :rtype: [tuple]"""
    forward = []
    backward = []
    for frame in range(len(sequence)):
        values = sequence.values(frame)
        if anchor in values:
            if values[anchor] > 0:
                forward.append(frame)
            elif values[anchor] < 0:
                backward.append(frame)
    return forward, backward

class MultiStartExperiment(Experiment):
    """The multistart experiment. The experiment works by utilizing anchor frames in the
    sequence. Anchor frames are frames where the given object is visible and can be used
    for initialization.

    The tracker is then initialized in each anchor frame and run until the end of the
    sequence either forward or backward.

    This experiment assumes that anchor frames are labeled in the sequence with a
    specific value (default is "anchor") and that the value of the object is positive
    for forward anchors and negative for backward anchors. If no anchor information is
    present in the sequence, the experiment will fail with an error. The experiment can
    be run with or without supervision.
    """

    anchor = String(default="anchor")

    def scan(self, tracker: Tracker, sequence: Sequence) -> tuple:
        """Scan the results of the experiment for the given tracker and sequence.

        :param tracker: The tracker to be scanned.
        :type tracker: Tracker
        :param sequence: The sequence to be scanned.
        :type sequence: Sequence

        :returns: A tuple containing three elements. The first element is a boolean indicating whether the experiment is complete. The second element is a list of files that are present. The third element is the results object.
        :rtype: [tuple]"""
    
        files = []
        complete = True

        results = self.results(tracker, sequence)

        forward, backward = find_anchors(sequence, self.anchor)

        if len(forward) == 0 and len(backward) == 0:
            raise RuntimeError("Sequence does not contain any anchors")

        for i in forward + backward:
            name = f"{sequence.name}_{i:08d}"
            if Trajectory.exists(results, name):
                files.extend(Trajectory.gather(results, name))
            else:
                complete = False

        return complete, files, results

    def execute(self, tracker: Tracker, sequence: Sequence, force: bool = False, callback: Callable = None) -> None:
        """Execute the experiment for the given tracker and sequence.

        :param tracker: The tracker to be executed.
        :type tracker: Tracker
        :param sequence: The sequence to be executed.
        :type sequence: Sequence
        :param force: Force re-execution of the experiment. Defaults to False.
        :type force: bool, optional
        :param callback: A callback function that is called after each frame. Defaults to None.
        :type callback: Callable, optional

        :raises RuntimeError: If the sequence does not contain any anchors."""

        results = self.results(tracker, sequence)

        forward, backward = find_anchors(sequence, self.anchor)

        if len(forward) == 0 and len(backward) == 0:
            raise RuntimeError("Sequence does not contain any anchors")

        total = len(forward) + len(backward)
        current = 0

        with self._get_runtime(tracker, sequence) as runtime:

            for i, reverse in [(f, False) for f in forward] + [(f, True) for f in backward]:
                name = f"{sequence.name}_{i:08d}"

                if Trajectory.exists(results, name) and not force:
                    continue

                if reverse:
                    proxy = FrameMapSequence(sequence, list(reversed(range(0, i + 1))))
                else:
                    proxy = FrameMapSequence(sequence, list(range(i, len(sequence))))

                queries = [ObjectQuery(self._get_initialization(proxy, 0), {}, i)]
                status = runtime.run(proxy, queries)

                trajectory = Trajectory(len(proxy))

                trajectory.set(0, Special(Trajectory.INITIALIZATION), {"time": status.times[0][0]})
                for frame in range(1, len(proxy)):
                    trajectory.set(frame, status.objects[0][frame], {"time": status.times[0][frame]})

                trajectory.write(results, name)

                current = current + 1
                if  callback:
                    callback(current / total)
