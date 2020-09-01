#pylint: disable=W0223

from typing import Callable

from vot.dataset import Sequence
from vot.region import Special, calculate_overlap

from vot.experiment import Experiment
from vot.tracker import Tracker, Trajectory
from vot.utilities import alias
from vot.utilities.attributes import Boolean, Integer, Float, List, String

class MultiRunExperiment(Experiment):

    repetitions = Integer(val_min=1, default=1)
    early_stop = Boolean(default=True)

    def _can_stop(self, tracker: Tracker, sequence: Sequence):
        if not self.early_stop:
            return False
        trajectories = self.gather(tracker, sequence)
        if len(trajectories) < 3:
            return False

        for trajectory in trajectories[1:]:
            if not trajectory.equals(trajectories[0]):
                return False

        return True

    def scan(self, tracker: Tracker, sequence: Sequence):
        
        results = self.results(tracker, sequence)

        files = []
        complete = True

        for i in range(1, self.repetitions+1):
            name = "%s_%03d" % (sequence.name, i)
            if Trajectory.exists(results, name):
                files.extend(Trajectory.gather(results, name))
            elif self._can_stop(tracker, sequence):
                break
            else:
                complete = False
                break

        return complete, files, results

    def gather(self, tracker: Tracker, sequence: Sequence):
        trajectories = list()
        results = self.results(tracker, sequence)
        for i in range(1, self.repetitions+1):
            name = "%s_%03d" % (sequence.name, i)
            if Trajectory.exists(results, name):
                trajectories.append(Trajectory.read(results, name))
        return trajectories

@alias("UnsupervisedExperiment", "unsupervised")
class UnsupervisedExperiment(MultiRunExperiment):

    def execute(self, tracker: Tracker, sequence: Sequence, force: bool = False, callback: Callable = None):

        results = self.results(tracker, sequence)

        with self._get_runtime(tracker, sequence) as runtime:

            for i in range(1, self.repetitions+1):
                name = "%s_%03d" % (sequence.name, i)

                if Trajectory.exists(results, name) and not force:
                    continue

                if self._can_stop(tracker, sequence):
                    return

                trajectory = Trajectory(sequence.length)

                _, properties, elapsed = runtime.initialize(sequence.frame(0), self._get_initialization(sequence, 0))

                properties["time"] = elapsed

                trajectory.set(0, Special(Special.INITIALIZATION), properties)

                for frame in range(1, sequence.length):
                    region, properties, elapsed = runtime.update(sequence.frame(frame))

                    properties["time"] = elapsed

                    trajectory.set(frame, region, properties)

                trajectory.write(results, name)

                if callback:
                    callback(i / self.repetitions)

@alias("SupervisedExperiment", "supervised")
class SupervisedExperiment(MultiRunExperiment):

    skip_initialize = Integer(val_min=1, default=1)
    skip_tags = List(String(), default=[])
    failure_overlap = Float(val_min=0, val_max=1, default=0)

    def execute(self, tracker: Tracker, sequence: Sequence, force: bool = False, callback: Callable = None):

        results = self.results(tracker, sequence)

        with self._get_runtime(tracker, sequence) as runtime:

            for i in range(1, self.repetitions+1):
                name = "%s_%03d" % (sequence.name, i)

                if Trajectory.exists(results, name) and not force:
                    continue

                if self._can_stop(tracker, sequence):
                    return

                trajectory = Trajectory(sequence.length)

                frame = 0
                while frame < sequence.length:

                    _, properties, elapsed = runtime.initialize(sequence.frame(frame), self._get_initialization(sequence, frame))

                    properties["time"] = elapsed

                    trajectory.set(frame, Special(Special.INITIALIZATION), properties)

                    frame = frame + 1

                    while frame < sequence.length:

                        region, properties, elapsed = runtime.update(sequence.frame(frame))

                        properties["time"] = elapsed

                        if calculate_overlap(region, sequence.groundtruth(frame), sequence.size) <= self.failure_overlap:
                            trajectory.set(frame, Special(Special.FAILURE), properties)
                            frame = frame + self.skip_initialize
 
                            if self.skip_tags:
                                while frame < sequence.length:
                                    if not [t for t in sequence.tags(frame) if t in self.skip_tags]:
                                        break
                                    frame = frame + 1
                            break
                        else:
                            trajectory.set(frame, region, properties)
                        frame = frame + 1

                if  callback:
                    callback(i / self.repetitions)

                trajectory.write(results, name)
