#pylint: disable=W0223

from abc import ABC
from typing import Callable

from vot.dataset import Sequence
from vot.region import Special, calculate_overlap

from vot.experiment import Experiment
from vot.tracker import Tracker, Trajectory
from vot.utilities import to_number, to_logical

class MultiRunExperiment(Experiment, ABC):

    def __init__(self, identifier: str, workspace: "Workspace", repetitions=1, early_stop=True, **kwargs):
        super().__init__(identifier, workspace, **kwargs)
        self._repetitions = to_number(repetitions, min_n=1)
        self._early_stop = to_logical(early_stop)

    @property
    def repetitions(self):
        return self._repetitions

    def _can_stop(self, tracker: Tracker, sequence: Sequence):
        if not self._early_stop:
            return False
        trajectories = self.gather(tracker, sequence)
        if len(trajectories) < 3:
            return False

        for trajectory in trajectories[1:]:
            if not trajectory.equals(trajectories[0]):
                return False

        return True


    def scan(self, tracker: Tracker, sequence: Sequence):
        
        results = self.workspace.results(tracker, self, sequence)

        files = []
        complete = True

        for i in range(1, self._repetitions+1):
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
        results = self.workspace.results(tracker, self, sequence)
        for i in range(1, self._repetitions+1):
            name = "%s_%03d" % (sequence.name, i)
            if Trajectory.exists(results, name):
                trajectories.append(Trajectory.read(results, name))
        return trajectories
        
class UnsupervisedExperiment(MultiRunExperiment):

    def execute(self, tracker: Tracker, sequence: Sequence, force: bool = False, callback: Callable = None):

        results = self.workspace.results(tracker, self, sequence)

        for i in range(1, self._repetitions+1):
            name = "%s_%03d" % (sequence.name, i)

            if Trajectory.exists(results, name) and not force:
                continue

            if self._can_stop(tracker, sequence):
                return

            trajectory = Trajectory(sequence.length)

            with self._get_runtime(tracker, sequence) as runtime:
                _, properties, elapsed = runtime.initialize(sequence.frame(0), self._get_initialization(sequence, 0))

                properties["time"] = elapsed

                trajectory.set(0, Special(Special.INITIALIZATION), properties)

                for frame in range(1, sequence.length):
                    region, properties, elapsed = runtime.update(sequence.frame(frame))

                    properties["time"] = elapsed

                    trajectory.set(frame, region, properties)

            trajectory.write(results, name)

            if  callback:
                callback(i / self._repetitions)

class SupervisedExperiment(MultiRunExperiment):

    def __init__(self, identifier: str, workspace: "Workspace", skip_initialize=1, skip_tags=(), failure_overlap=0, **kwargs):
        super().__init__(identifier, workspace, **kwargs)
        self._skip_initialize = to_number(skip_initialize, min_n=1)
        self._skip_tags = tuple(skip_tags)
        self._failure_overlap = to_number(failure_overlap, min_n=0, max_n=1, conversion=float)

    @property
    def skip_initialize(self):
        return self._skip_initialize

    @property
    def skip_tags(self):
        return self._skip_tags

    @property
    def failure_overlap(self):
        return self._failure_overlap

    def execute(self, tracker: Tracker, sequence: Sequence, force: bool = False, callback: Callable = None):

        results = self.workspace.results(tracker, self, sequence)

        for i in range(1, self._repetitions+1):
            name = "%s_%03d" % (sequence.name, i)

            if Trajectory.exists(results, name) and not force:
                continue

            if self._can_stop(tracker, sequence):
                return

            trajectory = Trajectory(sequence.length)

            with self._get_runtime(tracker, sequence) as runtime:

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
                callback(i / self._repetitions)

            trajectory.write(results, name)
