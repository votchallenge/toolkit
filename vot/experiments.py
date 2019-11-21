

import os
import json
import glob

from abc import abstractmethod, ABC

from vot.dataset import Sequence
from vot.tracker import Tracker
from vot.workspace import Results

class Experiment(ABC):

    def __init__(self, identifier:str):
        self._identifier = identifier

    @property
    def identifier(self):
        return self._identifier

    @abstractmethod
    def execute(self, tracker: Tracker, sequence: Sequence, results: Results):
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
        # TODO
        pass

class UnsupervisedExperiment(MultiRunExperiment):

    def __init__(self, identifier, repetitions=1):
        super().__init__(identifier, repetitions)

    def execute(self, tracker: Tracker, sequence: Sequence, results: Results):
        # TODO
        pass

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

    def execute(self, tracker: Tracker, sequence: Sequence, results: Results):
        # TODO
        pass

class RealtimeExperiment(SupervisedExperiment):

    def __init__(self, identifier, repetitions=1, burnin=0, skip_initialize = 1, failure_overlap = 0):
        super().__init__(identifier, repetitions, burnin, skip_initialize, failure_overlap)

    def execute(self, tracker: Tracker, sequence: Sequence, results: Results):
        # TODO
        pass