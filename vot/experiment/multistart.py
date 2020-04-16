
from typing import Callable

from vot.dataset import Sequence
from vot.dataset.proxy import FrameMapSequence
from vot.region import Special

from vot.experiment import Experiment
from vot.tracker import Tracker, Trajectory
from vot.utilities import to_number

def find_anchors(sequence: Sequence, anchor="anchor"):
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

class MultiStartExperiment(Experiment):

    def __init__(self, identifier: str, workspace: "Workspace", anchor: str = "anchor", **kwargs):
        super().__init__(identifier, workspace, **kwargs)
        self._anchor = str(anchor)

    @property
    def anchor(self):
        return self._anchor

    def scan(self, tracker: Tracker, sequence: Sequence):
    
        files = []
        complete = True

        results = self.workspace.results(tracker, self, sequence)

        forward, backward = find_anchors(sequence, self._anchor)

        if len(forward) == 0 and len(backward) == 0:
            raise RuntimeError("Sequence does not contain any anchors")

        for i in forward + backward:
            name = "%s_%08d" % (sequence.name, i)
            if Trajectory.exists(results, name):
                files.extend(Trajectory.gather(results, name))
            else:
                complete = False

        return complete, files, results

    def execute(self, tracker: Tracker, sequence: Sequence, force: bool = False, callback: Callable = None):

        results = self.workspace.results(tracker, self, sequence)

        forward, backward = find_anchors(sequence, self._anchor)

        if len(forward) == 0 and len(backward) == 0:
            raise RuntimeError("Sequence does not contain any anchors")

        total = len(forward) + len(backward)
        current = 0

        for i, reverse in [(f, False) for f in forward] + [(f, True) for f in backward]:
            name = "%s_%08d" % (sequence.name, i)

            if Trajectory.exists(results, name) and not force:
                continue

            if reverse:
                proxy = FrameMapSequence(sequence, list(reversed(range(0, i + 1))))
            else:
                proxy = FrameMapSequence(sequence, list(range(i, sequence.length)))

            trajectory = Trajectory(proxy.length)

            with self._get_runtime(tracker, sequence) as runtime:
                _, properties, elapsed = runtime.initialize(proxy.frame(0), self._get_initialization(proxy, 0))

                properties["time"] = elapsed

                trajectory.set(0, Special(Special.INITIALIZATION), properties)

                for frame in range(1, proxy.length):
                    region, properties, elapsed = runtime.update(proxy.frame(frame))

                    properties["time"] = elapsed

                    trajectory.set(frame, region, properties)

            trajectory.write(results, name)

            current = current + 1
            if  callback:
                callback(current / total)
