
from typing import Callable

from vot.dataset import Sequence
from vot.dataset.proxy import FrameMapSequence
from vot.region import Special

from vot.experiment import Experiment
from vot.tracker import Tracker, Trajectory
from vot.utilities import alias
from vot.utilities.attributes import String

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

@alias("MultiStartExperiment", "multistart")
class MultiStartExperiment(Experiment):

    anchor = String(default="anchor")

    def scan(self, tracker: Tracker, sequence: Sequence):
    
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

    def execute(self, tracker: Tracker, sequence: Sequence, force: bool = False, callback: Callable = None):

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
