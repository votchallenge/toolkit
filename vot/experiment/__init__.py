

import os
import json
import glob

from abc import abstractmethod, ABC

class Experiment(ABC):

    def __init__(self, identifier:str):
        self._identifier = identifier

    @property
    def identifier(self):
        return self._identifier

    @abstractmethod
    def execute(self, tracker: "Tracker", sequence: "Sequence", results: "Results", force:bool=False):
        pass

    @abstractmethod
    def scan(self, tracker: "Tracker", sequence: "Sequence", results: "Results"):
        pass

from .multirun import UnsupervisedExperiment, SupervisedExperiment