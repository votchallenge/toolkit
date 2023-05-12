

import logging
import typing
from datetime import datetime
from abc import abstractmethod

from class_registry import ClassRegistry

from attributee import Attributee, Object, Integer, Float, Nested, List

from vot.tracker import TrackerException
from vot.utilities import Progress, to_number, import_class

experiment_registry = ClassRegistry("vot_experiment")
transformer_registry = ClassRegistry("vot_transformer")

class RealtimeConfig(Attributee):
    """Config proxy for real-time experiment.
    """

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

    storage = context.parent.storage.substorage("cache").substorage("transformer")

    if typename in transformer_registry:
        transformer = transformer_registry.get(typename, cache=storage, **kwargs)
        assert isinstance(transformer, Transformer)
        return transformer
    else:
        transformer_class = import_class(typename)
        assert issubclass(transformer_class, Transformer)
        return transformer_class(cache=storage, **kwargs)

def analysis_resolver(typename, context, **kwargs):
    from vot.analysis import Analysis, analysis_registry

    if typename in analysis_registry:
        analysis = analysis_registry.get(typename, **kwargs)
        assert isinstance(analysis, Analysis)
    else:
        analysis_class = import_class(typename)
        assert issubclass(analysis_class, Analysis)
        analysis = analysis_class(**kwargs)

    assert analysis.compatible(context.parent)

    return analysis

class Experiment(Attributee):
    """Experiment abstract base class. 

    """

    UNKNOWN = 0
    INITIALIZATION = 1

    realtime = Nested(RealtimeConfig, default=None, description="Realtime modifier config")
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
    def _multiobject(self) -> bool:
        # TODO: at some point this may be a property for all experiments
        return False

    @property
    def storage(self) -> "Storage":
        return self._storage

    def _get_initialization(self, sequence: "Sequence", index: int, id: str = None):
        if not self._multiobject and id is None:
            return sequence.groundtruth(index)
        else:
            return sequence.frame(index).object(id)

    def _get_runtime(self, tracker: "Tracker", sequence: "Sequence", multiobject=False):
        from ..tracker import SingleObjectTrackerRuntime, RealtimeTrackerRuntime, MultiObjectTrackerRuntime

        runtime = tracker.runtime()

        if multiobject:
            if not runtime.multiobject:
                raise TrackerException("Tracker {} does not support multi-object experiments".format(tracker.identifier))
                #runtime = MultiObjectTrackerRuntime(runtime)
        else:
            runtime = SingleObjectTrackerRuntime(runtime)

        if not self.realtime is None:
            grace = to_number(self.realtime.grace, min_n=0)
            fps = to_number(self.realtime.fps, min_n=0, conversion=float)
            interval = 1 / float(sequence.metadata("fps", fps))
            runtime = RealtimeTrackerRuntime(runtime, grace, interval)

        return runtime

    @abstractmethod
    def execute(self, tracker: "Tracker", sequence: "Sequence", force: bool = False, callback: typing.Callable = None):
        raise NotImplementedError

    @abstractmethod
    def scan(self, tracker: "Tracker", sequence: "Sequence"):
        """ Scan results for a given tracker and sequence. """
        raise NotImplementedError

    def results(self, tracker: "Tracker", sequence: "Sequence") -> "Results":
        if tracker.storage is not None:
            return tracker.storage.results(tracker, self, sequence)
        return self._storage.results(tracker, self, sequence)

    def log(self, identifier: str):
        return self._storage.substorage("logs").write("{}_{:%Y-%m-%dT%H-%M-%S.%f%z}.log".format(identifier, datetime.now()))

    def transform(self, sequences):
        from vot.dataset import Sequence
        from vot.experiment.transformer import SingleObject
        if isinstance(sequences, Sequence):
            sequences = [sequences]
        
        transformers = list(self.transformers)

        if not self._multiobject:
            transformers.insert(0, SingleObject())

        # Process sequences one transformer at the time. The number of sequences may grow
        for transformer in transformers:
            transformed = []
            for sequence in sequences:
                transformed.extend(transformer(sequence))
            sequences = transformed

        return sequences

from .multirun import UnsupervisedExperiment, SupervisedExperiment
from .multistart import MultiStartExperiment

def run_experiment(experiment: Experiment, tracker: "Tracker", sequences: typing.List["Sequence"], force: bool = False, persist: bool = False):
    """A helper function that performs a given experiment with a given tracker on a list of sequences.

    Args:
        experiment (Experiment): The experiment object
        tracker (Tracker): The tracker object
        sequences (typing.List[Sequence]): List of sequences.
        force (bool, optional): Ignore the cached results, rerun all the experiments. Defaults to False.
        persist (bool, optional): Continue runing even if exceptions were raised. Defaults to False.

    Raises:
        TrackerException: If the experiment is interrupted
    """

    class EvaluationProgress(object):

        def __init__(self, description, total):
            self.bar = Progress(description, total)
            self._finished = 0

        def __call__(self, progress):
            self.bar.absolute(self._finished + min(1, max(0, progress)))

        def push(self):
            self._finished = self._finished + 1
            self.bar.absolute(self._finished)

        def close(self):
            self.bar.close()

    logger = logging.getLogger("vot")

    transformed = []
    for sequence in sequences:
        transformed.extend(experiment.transform(sequence))
    sequences = transformed

    progress = EvaluationProgress("{}/{}".format(tracker.identifier, experiment.identifier), len(sequences))
    for sequence in sequences:
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
                raise TrackerException("Experiment interrupted", te, tracker=tracker)
        progress.push()

    progress.close()