
from vot.dataset import Sequence
from vot.dataset.proxy import FrameMapSequence
from vot.region import Special

from vot.experiment import Experiment
from vot.tracker import Tracker, Trajectory, Results

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

    def __init__(self, identifier: str, anchor: str = "anchor"):
        super().__init__(identifier)
        self._anchor = anchor

    def scan(self, tracker: Tracker, sequence: Sequence, results: Results):
        
        files = []
        complete = True

        forward, backward = find_anchors(sequence, self._anchor)

        if len(forward) == 0 and len(backward) == 0:
            raise RuntimeError("Sequence does not contain any anchors")

        for i in forward + backward:
            name = "%s_%08d" % (sequence.name, i)
            if Trajectory.exists(results, name):
                files.extend(Trajectory.gather(results, name))
            else:
                complete = False

        return complete, files

    def execute(self, tracker: Tracker, sequence: Sequence, results: Results, force: bool = False):

        forward, backward = find_anchors(sequence, self._anchor)

        if len(forward) == 0 and len(backward) == 0:
            raise RuntimeError("Sequence does not contain any anchors")

        for i, reverse in [(f, False) for f in forward] + [(f, True) for f in backward]:
            name = "%s_%08d" % (sequence.name, i)

            if Trajectory.exists(results, name) and not force:
                continue

            if reverse:
                proxy = FrameMapSequence(sequence, list(reversed(range(i, sequence.length))))
            else:
                proxy = FrameMapSequence(sequence, list(range(i, sequence.length)))

            trajectory = Trajectory(proxy.length)

            with tracker.runtime() as runtime:
                _, properties, elapsed = runtime.initialize(proxy.frame(0), proxy.groundtruth(0))

                properties["time"] = elapsed

                trajectory.set(0, Special(Special.INITIALIZATION), properties)

                for frame in range(1, proxy.length):
                    region, properties, elapsed = runtime.update(proxy.frame(frame))

                    properties["time"] = elapsed

                    trajectory.set(frame, region, properties)

            trajectory.write(results, name)