
from vot.dataset import Sequence
from vot.dataset.proxy import FrameMapSequence
from vot.region import Special

from vot.experiment import Experiment
from vot.tracker import Tracker, Trajectory, Results

def find_tag(sequence: Sequence, tag: str):
    found = []
    for frame in range(sequence.length):
        if tag in sequence.tags(frame):
            found.append(tag)
    return found


class MultiStartExperiment(Experiment):

    def __init__(self, identifier: str, anchor_tag: str = "anchor"):
        super().__init__(identifier)
        self._anchor_tag = anchor_tag

    def scan(self, tracker: Tracker, sequence: Sequence, results: Results):
        
        files = []
        complete = True

        anchors = find_tag(sequence, self._anchor_tag)

        if len(anchors) == 0:
            raise RuntimeError("Sequence does not contain any anchors")

        for i in anchors:
            name = "%s_%08d" % (sequence.name, i)
            if Trajectory.exists(results, name):
                files.extend(Trajectory.gather(results, name))
            else:
                complete = False

        return complete, files

    def execute(self, tracker: Tracker, sequence: Sequence, results: Results, force: bool = False):

        anchors = find_tag(sequence, self._anchor_tag)

        if len(anchors) == 0:
            raise RuntimeError("Sequence does not contain any anchors")

        for i in anchors:
            name = "%s_%08d" % (sequence.name, i)

            if Trajectory.exists(results, name) and not force:
                continue

            if i > sequence.length / 2:
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