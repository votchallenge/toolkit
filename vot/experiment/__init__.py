

import os
import json
import glob
import logging
import typing
from datetime import datetime

from abc import abstractmethod, ABC

from vot.tracker import RealtimeTrackerRuntime, TrackerException
from vot.utilities import Progress, to_number, import_class
from vot.utilities.attributes import Attributee, Object, Integer, Float, Nested, List

class RealtimeConfig(Attributee):

    grace = Integer(val_min=0, default=0)
    fps = Float(val_min=0, default=20)

class NoiseConfig(Attributee):
    # Not implemented yet
    placeholder = Integer(default=1)

class InjectConfig(Attributee):
    # Not implemented yet
    placeholder = Integer(default=1)

def transformer_resolver(typename, context, **kwargs):
    from vot.experiment.transformer import Transformer
    transformer_class = import_class(typename)
    assert issubclass(transformer_class, Transformer)

    if "parent" in context:
        storage = context["parent"].storage.substorage("cache").substorage("transformer")
    else:
        storage = None

    return transformer_class(cache=storage, **kwargs)

def analysis_resolver(typename, context, **kwargs):
    from vot.analysis import Analysis
    analysis_class = import_class(typename)
    assert issubclass(analysis_class, Analysis)

    analysis = analysis_class(**kwargs)

    if "parent" in context:
        assert analysis.compatible(context["parent"])

    return analysis

class Experiment(Attributee):

    realtime = Nested(RealtimeConfig, default=None)
    noise = Nested(NoiseConfig, default=None)
    inject = Nested(InjectConfig, default=None)
    transformers = List(Object(transformer_resolver), default=[])
    analyses = List(Object(analysis_resolver), default=[])

    def __init__(self, _identifier: str, _storage: "LocalStorage", **kwargs):
        self._identifier = _identifier
        self._storage = _storage
        super().__init__(**kwargs)
        # TODO: validate analysis names

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def storage(self) -> "Storage":
        return self._storage

    def _get_initialization(self, sequence: "Sequence", index: int):
        return sequence.groundtruth(index)

    def _get_runtime(self, tracker: "Tracker", sequence: "Sequence"):
        if not self.realtime is None:
            grace = to_number(self.realtime.grace, min_n=0)
            fps = to_number(self.realtime.fps, min_n=0, conversion=float)
            interval = 1 / float(sequence.metadata("fps", fps))
            runtime = RealtimeTrackerRuntime(tracker.runtime(), grace, interval)
        else:
            runtime = tracker.runtime()
        return runtime

    @abstractmethod
    def execute(self, tracker: "Tracker", sequence: "Sequence", force: bool = False, callback: typing.Callable = None):
        raise NotImplementedError

    @abstractmethod
    def scan(self, tracker: "Tracker", sequence: "Sequence"):
        raise NotImplementedError

    def results(self, tracker: "Tracker", sequence: "Sequence") -> "Results":
        return self._storage.results(tracker, self, sequence)

    def log(self, identifier: str):
        return self._storage.substorage("logs").write("{}_{:%Y-%m-%dT%H-%M-%S.%f%z}.log".format(identifier, datetime.now()))

    def transform(self, sequence: "Sequence"):
        for transformer in self.transformers:
            sequence = transformer(sequence)
        return sequence

from .multirun import UnsupervisedExperiment, SupervisedExperiment
from .multistart import MultiStartExperiment

def run_experiment(experiment: Experiment, tracker: "Tracker", sequences: typing.List["Sequence"], force: bool = False, persist: bool = False):

    class EvaluationProgress(object):

        def __init__(self, description, total):
            self.bar = Progress(description, total)
            self._finished = 0

        def __call__(self, progress):
            self.bar.absolute(self._finished + min(1, max(0, progress)))

        def push(self):
            self._finished = self._finished + 1
            self.bar.absolute(self._finished)

    logger = logging.getLogger("vot")

    progress = EvaluationProgress("{}/{}".format(tracker.identifier, experiment.identifier), len(sequences))
    for sequence in sequences:
        sequence = experiment.transform(sequence)
        try:
            experiment.execute(tracker, sequence, force=force, callback=progress)
        except TrackerException as te:
            logger.error("Tracker %s encountered an error: %s", te.tracker.identifier, te)
            logger.debug(te, exc_info=True)
            if not te.log is None:
                with experiment.log(te.tracker.identifier) as flog:
                    flog.write(te.log)
                    logger.error("Tracker output written to file: %s", flog.name)
            if not persist:
                raise te
        progress.push()

