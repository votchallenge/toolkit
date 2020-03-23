

import os
import json
import glob
import logging

from typing import Callable

from abc import abstractmethod, ABC

from vot.tracker import RealtimeTrackerRuntime, TrackerException
from vot.utilities import Progress, to_number

class Experiment(ABC):

    def __init__(self, identifier: str, workspace: "Workspace", realtime: dict = None, noise: dict = None, inject: dict = None):
        super().__init__()
        self._identifier = identifier
        self._workspace = workspace
        self._realtime = realtime
        self._noise = noise # Not implemented yet
        self._inject = inject # Not implemented yet

    @property
    def workspace(self) -> "Workspace":
        return self._workspace

    @property
    def identifier(self) -> str:
        return self._identifier

    def _get_initialization(self, sequence: "Sequence", index: int):
        return sequence.groundtruth(index)

    def _get_runtime(self, tracker: "Tracker", sequence: "Sequence"):
        if not self._realtime is None:
            grace = to_number(self._realtime.get("grace", 0), min_n=0)
            fps = to_number(self._realtime.get("fps", 20), min_n=0, conversion=float)
            interval = 1 / float(sequence.metadata("fps", fps))
            runtime = RealtimeTrackerRuntime(tracker.runtime(), grace, interval)
        else:
            runtime = tracker.runtime()
        return runtime

    @abstractmethod
    def execute(self, tracker: "Tracker", sequence: "Sequence", force: bool = False, callback: Callable = None):
        pass

    @abstractmethod
    def scan(self, tracker: "Tracker", sequence: "Sequence"):
        pass

    def results(self, tracker: "Tracker", sequence: "Sequence") -> "Results":
        return self._workspace.results(tracker, self, sequence)

from .multirun import UnsupervisedExperiment, SupervisedExperiment
from .multistart import MultiStartExperiment

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
        transformers = experiment.workspace.stack.transformers(experiment)
        for transformer in transformers:
            sequence = transformer(sequence)
        try:
            experiment.execute(tracker, sequence, force=force, callback=progress)
        except TrackerException as te:
            logger.error("Tracker %s encountered an error: %s", te.tracker.identifier, te)
            logger.debug(te, exc_info=True)
            if not te.log is None:
                with experiment.workspace.open_log(te.tracker.identifier) as flog:
                    flog.write(te.log)
                    logger.error("Tracker output writtent to file: %s", flog.name)
            if not persist:
                raise te
        progress.push()

