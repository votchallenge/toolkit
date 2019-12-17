
import os
import json
import glob

from vot.dataset import Sequence
from vot.region import Special, Region, calculate_overlap

from vot.experiment import Experiment
from vot.tracker import Tracker, Trajectory, Results

class MultiRunExperiment(Experiment):

    def __init__(self, identifier:str, repetitions=1):
        super().__init__(identifier)
        self._repetitions = repetitions

    @property
    def repetitions(self):
        return self._repetitions

    def scan(self, tracker: Tracker, sequence: Sequence, results: Results):
        
        files = []
        complete = True

        for i in range(1, self._repetitions+1):
            name = "%s_%03d" % (sequence.name, i)
            if Trajectory.exists(results, name):
                files.extend(Trajectory.gather(results, name))
            else:
                complete = False

        return complete, files

class UnsupervisedExperiment(MultiRunExperiment):

    def __init__(self, identifier, repetitions=1):
        super().__init__(identifier, repetitions)

    def execute(self, tracker: Tracker, sequence: Sequence, results: Results, force:bool=False):

        for i in range(1, self._repetitions+1):
            name = "%s_%03d" % (sequence.name, i)

            if Trajectory.exists(results, name) and not force:
                continue

            trajectory = Trajectory(sequence.length)

            with tracker.runtime() as runtime:
                _, properties, elapsed = runtime.initialize(sequence.frame(0), sequence.groundtruth(0))

                properties["time"] = elapsed

                trajectory.set(0, Special(Special.INITIALIZATION), properties)

                for frame in range(1, sequence.length):
                    region, properties, elapsed = runtime.update(sequence.frame(frame))

                    properties["time"] = elapsed

                    trajectory.set(frame, region, properties)

            trajectory.write(results, name)

class SupervisedExperiment(MultiRunExperiment):

    def __init__(self, identifier, repetitions=1, skip_initialize = 1, skip_tags = (), failure_overlap = 0):
        super().__init__(identifier, repetitions)
        self._skip_initialize = skip_initialize
        self._skip_tags = skip_tags
        self._failure_overlap = failure_overlap

    @property
    def skip_initialize(self):
        return self._skip_initialize

    @property
    def skip_tags(self):
        return self._skip_tags

    @property
    def failure_overlap(self):
        return self._failure_overlap

    def execute(self, tracker: Tracker, sequence: Sequence, results: Results, force:bool=False):

        for i in range(1, self._repetitions+1):
            name = "%s_%03d" % (sequence.name, i)

            if Trajectory.exists(results, name) and not force:
                continue

            trajectory = Trajectory(sequence.length)

            with tracker.runtime() as runtime:

                start = 0
                while start < sequence.length:

                    _, properties, elapsed = runtime.initialize(sequence.frame(start), sequence.groundtruth(start))

                    properties["time"] = elapsed

                    trajectory.set(start, Special(Special.INITIALIZATION), properties)

                    for frame in range(start+1, sequence.length):
                        region, properties, elapsed = runtime.update(sequence.frame(frame))

                        properties["time"] = elapsed

                        if calculate_overlap(region, sequence.groundtruth(frame), sequence.size) <= self.failure_overlap:
                            trajectory.set(frame, Special(Special.FAILURE), properties)
                            runtime.restart()
                            start = frame + self.skip_initialize
                            if self.skip_tags:
                                while start < sequence.length:
                                    if not [t for t in sequence.tags(start) if t in self.skip_tags]:
                                        break
                            break
                        else:
                            trajectory.set(frame, region, properties)

            trajectory.write(results, name)

class RealtimeExperiment(SupervisedExperiment):

    def __init__(self, identifier, repetitions=1, burnin=0, skip_initialize = 1, failure_overlap = 0):
        super().__init__(identifier, repetitions, burnin, skip_initialize, failure_overlap)

    def execute(self, tracker: Tracker, sequence: Sequence, results: Results, force:bool=False):
        # TODO
        pass