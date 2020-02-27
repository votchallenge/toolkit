

import os
import json
import glob
import logging

from typing import Callable

from abc import abstractmethod, ABC

from vot.tracker import RealtimeTrackerRuntime, TrackerException
from vot.utilities import Progress, to_number

class Experiment(ABC):

    def __init__(self, identifier: str, workspace: "Workspace"):
        super().__init__()
        self._identifier = identifier
        self._workspace = workspace

    @property
    def workspace(self) -> "Workspace":
        return self._workspace

    @property
    def identifier(self) -> str:
        return self._identifier

    def _get_runtime(self, tracker: "Tracker", sequence: "Sequence"):
        return tracker.runtime()

    @abstractmethod
    def execute(self, tracker: "Tracker", sequence: "Sequence", force: bool = False, callback: Callable = None):
        pass

    @abstractmethod
    def scan(self, tracker: "Tracker", sequence: "Sequence"):
        pass

    def results(self, tracker: "Tracker", sequence: "Sequence") -> "Results":
        return self._workspace.results(tracker, self, sequence)

class RealtimeMixin(Experiment):

    def __init__(self, identifier: str, workspace: "Workspace", grace:int = 1):
        super().__init__(identifier, workspace)
        self._grace = to_number(grace, min_n=0)

    def _get_runtime(self, tracker: "Tracker", sequence: "Sequence"):
        interval = 1 / float(sequence.metadata("fps", 20))

        return RealtimeTrackerRuntime(super()._get_runtime(tracker, sequence), self._grace, interval)

from .multirun import UnsupervisedExperiment, SupervisedExperiment, RealtimeSupervisedExperiment, RealtimeUnsupervisedExperiment

from .multistart import MultiStartExperiment, RealtimeMultiStartExperiment

class EvaluationProgress(object):

    def __init__(self, description, total):
        self.bar = Progress(desc=description, total=total, unit="sequence")
        self._finished = 0

    def __call__(self, progress):
        self.bar.update_absolute(self._finished + min(1, max(0, progress)))

    def push(self):
        self._finished = self._finished + 1
        self.bar.update_absolute(self._finished)

def run_experiment(experiment: Experiment, tracker: "Tracker", force: bool = False, persist: bool = False):

    logger = logging.getLogger("vot")

    progress = EvaluationProgress("{}/{}".format(tracker.identifier, experiment.identifier), len(experiment.workspace.dataset))
    for sequence in experiment.workspace.dataset:
        try:
            experiment.execute(tracker, sequence, force=force, callback=progress)
        except TrackerException as te:
            logger.error("Tracker {} encountered an error: {}".format(te.tracker.identifier, te))
            if not te.log is None:
                with experiment.workspace.open_log(te.tracker.identifier) as flog:
                    flog.write(te.log)
                    logger.error("Tracker output writtent to file: {}".format(flog.name))
            if not persist:
                raise te
        progress.push()

