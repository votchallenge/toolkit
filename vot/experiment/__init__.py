

import os
import json
import glob

from abc import abstractmethod, ABC

class Experiment(ABC):

    def __init__(self, identifier: str, workspace: "Workspace"):
        self._identifier = identifier
        self._workspace = workspace

    @property
    def workspace(self) -> "Workspace":
        return self._workspace

    @property
    def identifier(self) -> str:
        return self._identifier

    @abstractmethod
    def execute(self, tracker: "Tracker", sequence: "Sequence", force: bool = False):
        pass

    @abstractmethod
    def scan(self, tracker: "Tracker", sequence: "Sequence"):
        pass

    def results(self, tracker: "Tracker", sequence: "Sequence") -> "Results":
        return self._workspace.results(tracker, self, sequence)

from .multirun import UnsupervisedExperiment, SupervisedExperiment

from .multistart import MultiStartExperiment