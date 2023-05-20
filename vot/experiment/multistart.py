
""" This module implements the multistart experiment. """

from typing import Callable

from vot.dataset import Sequence
from vot.dataset.proxy import FrameMapSequence
from vot.region import Special

from attributee import String

from vot.experiment import Experiment, experiment_registry
from vot.tracker import Tracker, Trajectory

def find_anchors(sequence: Sequence, anchor="anchor"):
    """Find anchor frames in the sequence. Anchor frames are frames where the given object is visible and can be used for initialization.
    
    Args:
        sequence (Sequence): The sequence to be scanned.
        anchor (str, optional): The name of the object to be used as an anchor. Defaults to "anchor".
        
    Returns:
        [tuple]: A tuple containing two lists of frames. The first list contains forward anchors, the second list contains backward anchors.
    """
    forward = []
    backward = []
    for frame in range(sequence.length):
        values = sequence.values(frame)
        if anchor in values:
            if values[anchor] > 0:
                forward.append(frame)
            elif values[anchor] < 0:
                backward.append(frame)
    return forward, backward

@experiment_registry.register("multistart")
class MultiStartExperiment(Experiment):
    """The multistart experiment. The experiment works by utilizing anchor frames in the sequence. 
    Anchor frames are frames where the given object is visible and can be used for initialization. 
    The tracker is then initialized in each anchor frame and run until the end of the sequence either forward or backward. 
    """

    anchor = String(default="anchor")

    def scan(self, tracker: Tracker, sequence: Sequence) -> tuple:
        """Scan the results of the experiment for the given tracker and sequence.
        
        Args:
            tracker (Tracker): The tracker to be scanned.
            sequence (Sequence): The sequence to be scanned.
        
        Returns:
            [tuple]: A tuple containing three elements. The first element is a boolean indicating whether the experiment is complete. The second element is a list of files that are present. The third element is the results object."""
    
        files = []
        complete = True

        results = self.results(tracker, sequence)

        forward, backward = find_anchors(sequence, self.anchor)

        if len(forward) == 0 and len(backward) == 0:
            raise RuntimeError("Sequence does not contain any anchors")

        for i in forward + backward:
            name = "%s_%08d" % (sequence.name, i)
            if Trajectory.exists(results, name):
                files.extend(Trajectory.gather(results, name))
            else:
                complete = False

        return complete, files, results

    def execute(self, tracker: Tracker, sequence: Sequence, force: bool = False, callback: Callable = None) -> None:
        """Execute the experiment for the given tracker and sequence.
        
        Args:
            tracker (Tracker): The tracker to be executed.
            sequence (Sequence): The sequence to be executed.
            force (bool, optional): Force re-execution of the experiment. Defaults to False.
            callback (Callable, optional): A callback function that is called after each frame. Defaults to None.
            
        Raises:
            RuntimeError: If the sequence does not contain any anchors.
        """

        results = self.results(tracker, sequence)

        forward, backward = find_anchors(sequence, self.anchor)

        if len(forward) == 0 and len(backward) == 0:
            raise RuntimeError("Sequence does not contain any anchors")

        total = len(forward) + len(backward)
        current = 0

        with self._get_runtime(tracker, sequence) as runtime:

            for i, reverse in [(f, False) for f in forward] + [(f, True) for f in backward]:
                name = "%s_%08d" % (sequence.name, i)

                if Trajectory.exists(results, name) and not force:
                    continue

                if reverse:
                    proxy = FrameMapSequence(sequence, list(reversed(range(0, i + 1))))
                else:
                    proxy = FrameMapSequence(sequence, list(range(i, sequence.length)))

                trajectory = Trajectory(proxy.length)

                _, elapsed = runtime.initialize(proxy.frame(0), self._get_initialization(proxy, 0))

                trajectory.set(0, Special(Trajectory.INITIALIZATION), {"time": elapsed})

                for frame in range(1, proxy.length):
                    object, elapsed = runtime.update(proxy.frame(frame))

                    object.properties["time"] = elapsed

                    trajectory.set(frame, object.region, object.properties)

                trajectory.write(results, name)

                current = current + 1
                if  callback:
                    callback(current / total)
