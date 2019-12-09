

import os
import json
import glob

from abc import abstractmethod, ABC

from vot.dataset import Sequence
from vot.tracker import Tracker, Trajectory
from vot.workspace import Results
from vot.region import Special, Region

class Experiment(ABC):

    def __init__(self, identifier:str):
        self._identifier = identifier

    @property
    def identifier(self):
        return self._identifier

    @abstractmethod
    def execute(self, tracker: Tracker, sequence: Sequence, results: Results, force:bool=False):
        pass

    @abstractmethod
    def scan(self, tracker: Tracker, sequence: Sequence, results: Results):
        pass


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
            name = "%s_%03d.txt" % (sequence.name, i)
            if results.exists(name):
                files.append(name)
            else:
                complete = False

        name = "%s_time.txt" % (sequence.name)
        if results.exists(name):
            files.append(name)
        else:
            complete = False

        return complete, files

class UnsupervisedExperiment(MultiRunExperiment):

    def __init__(self, identifier, repetitions=1):
        super().__init__(identifier, repetitions)

    def execute(self, tracker: Tracker, sequence: Sequence, results: Results, force:bool=False):

        for i in range(1, self._repetitions+1):
            name = "%s_%03d" % (sequence.name, i)

            if force or not Trajectory.exists(results, name):
                continue

            runtime = tracker.runtime()

            trajectory = Trajectory(sequence.length)

            _, properties = runtime.initialize(sequence.frame(0), sequence.groundtruth(0))

            trajectory.set(0, Special(Special.INITIALIZATION), properties)

            for frame in range(start=1, end=sequence.length):
                region, properties = runtime.update(sequence.frame(frame))

                trajectory.set(frame, region, properties)

            trajectory.write(results, name)

class SupervisedExperiment(MultiRunExperiment):

    def __init__(self, identifier, repetitions=1, burnin=0, skip_initialize = 1, failure_overlap = 0):
        super().__init__(identifier, repetitions)
        self._burnin = burnin
        self._skip_initialize = skip_initialize
        self._failure_overlap = failure_overlap

    @property
    def skip_initialize(self):
        return self._skip_initialize

    @property
    def burnin(self):
        return self._burnin

    @property
    def failure_overlap(self):
        return self._failure_overlap

    def execute(self, tracker: Tracker, sequence: Sequence, results: Results, force:bool=False):
        # TODO
        pass

class RealtimeExperiment(SupervisedExperiment):

    def __init__(self, identifier, repetitions=1, burnin=0, skip_initialize = 1, failure_overlap = 0):
        super().__init__(identifier, repetitions, burnin, skip_initialize, failure_overlap)

    def execute(self, tracker: Tracker, sequence: Sequence, results: Results, force:bool=False):
        # TODO
        pass