"""Experiments are the main building blocks of the toolkit.

They are used to evaluate trackers on sequences in various ways.
"""

import typing
from datetime import datetime
from abc import abstractmethod

from attributee import Attributee, Object, Integer, Float, Nested, List, Boolean

from vot.tracker import TrackerException, ObjectStatus
from vot.utilities import Progress, to_number, Registry
from vot.dataset.proxy import IgnoreSpecialObjects

class RealtimeConfig(Attributee):
    """Config proxy for real-time experiment."""

    grace = Integer(val_min=0, default=0)
    fps = Float(val_min=0, default=20)

class NoiseConfig(Attributee):
    """Config proxy for noise modifiers in experiments."""
    # Not implemented yet
    placeholder = Integer(default=1)

class InjectConfig(Attributee):
    """Config proxy for parameter injection in experiments."""
    # Not implemented yet
    placeholder = Integer(default=1)

def transformer_resolver(typename, context, **kwargs):
    """Resolve a transformer from a string. If the transformer is not registered, it is
    imported as a class and instantiated with the provided arguments.

    :param typename: Name of the transformer
    :type typename: str
    :param context: Context of the resolver
    :type context: Attributee

    :returns: Resolved transformer
    :rtype: Transformer"""
    from vot.utilities import import_class
    from vot.experiment.transformer import Transformer


    if context.parent.storage is None:
        storage = None
    else:
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
    """Resolve an analysis from a string. If the analysis is not registered, it is
    imported as a class and instantiated with the provided arguments.

    :param typename: Name of the analysis
    :type typename: str
    :param context: Context of the resolver
    :type context: Attributee

    :returns: Resolved analysis
    :rtype: Analysis"""
    from vot.utilities import import_class
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

    Each experiment is responsible for running a tracker on a sequence and storing
    results into dedicated storage.
    """

    UNKNOWN = 0
    INITIALIZATION = 1

    realtime = Nested(RealtimeConfig, default=None, description="Realtime modifier config")
    noise = Nested(NoiseConfig, default=None)
    inject = Nested(InjectConfig, default=None)
    transformers = List(Object(transformer_resolver), default=[])
    analyses = List(Object(analysis_resolver), default=[])
    ignore_special = Boolean(default=True, description="Ignore special objects in experiment")

    def __init__(self, _identifier: str, _storage: "Storage", **kwargs):
        """Initialize an experiment.

        :param _identifier: Identifier of the experiment
        :type _identifier: str
        :param _storage: Storage to use for storing results
        :type _storage: Storage

        :param **kwargs: Additional arguments

        :raises ValueError: If the identifier is not valid"""
        self._identifier = _identifier
        self._storage = _storage
        super().__init__(**kwargs)
        # TODO: validate analysis names

    @property
    def storage(self) -> "Storage":
        """Storage to use for storing results. Can be None if the experiment is not
        supposed to store results.

        :returns: Storage to use for storing results
        :rtype: Storage"""
        return self._storage

    @property
    def identifier(self) -> str:
        """Identifier of the experiment.

        :returns: Identifier of the experiment
        :rtype: str"""
        return self._identifier

    @property
    def _multiobject(self) -> bool:
        """Whether the experiment is multi-object or not.

        :returns: Whether the experiment is multi-object or not
        :rtype: bool"""
        # TODO: at some point this may be a property for all experiments
        return False

    def _get_initialization(self, sequence: "Sequence", index: int, oid: str = None) -> ObjectStatus:
        """Get initialization for a given sequence, index and object id.

        :param sequence: Sequence to get initialization for
        :type sequence: Sequence
        :param index: Index of the frame to get initialization for
        :type index: int
        :param id: Object id to get initialization for
        :type id: str

        :returns: Initialization state for the given sequence, index and object id
        :rtype: Initialization
        :raises ValueError: If the sequence does not contain the given index or object id"""
        if not self._multiobject and oid is None:
            return ObjectStatus(sequence.groundtruth(index), {})
        else:
            return ObjectStatus(sequence.frame(index).object(oid), {})

    def _get_runtime(self, tracker: "Tracker", sequence: "Sequence", multiobject=False):
        """Get runtime for a given tracker and sequence. Can convert single-object
        runtimes to multi-object runtimes.

        :param tracker: Tracker to get runtime for
        :type tracker: Tracker
        :param sequence: Sequence to get runtime for
        :type sequence: Sequence
        :param multiobject: Whether the runtime should be multi-object or not
        :type multiobject: bool

        :returns: Runtime for the given tracker and sequence
        :rtype: TrackerRuntime
        :raises TrackerException: If the tracker does not support multi-object experiments"""
        from vot.tracker import RealtimeTrackerRuntime

        runtime = tracker.runtime()

        if not self.realtime is None:
            grace = to_number(self.realtime.grace, min_n=0)
            fps = to_number(self.realtime.fps, min_n=0, conversion=float)
            interval = 1 / float(sequence.metadata("fps", fps))
            runtime = RealtimeTrackerRuntime(runtime, grace, interval)

        return runtime

    @abstractmethod
    def execute(self, tracker: "Tracker", sequence: "Sequence", force: bool = False, callback: typing.Callable = None):
        """Execute the experiment for a given tracker and sequence.

        :param tracker: Tracker to execute
        :type tracker: Tracker
        :param sequence: Sequence to execute
        :type sequence: Sequence
        :param force: Whether to force execution even if the results are already present
        :type force: bool
        :param callback: Callback to call after each frame
        :type callback: typing.Callable

        :returns: Results for the tracker and sequence
        :rtype: Results"""
        raise NotImplementedError

    @abstractmethod
    def scan(self, tracker: "Tracker", sequence: "Sequence"):
        """Scan results for a given tracker and sequence.

        :param tracker: Tracker to scan results for
        :type tracker: Tracker
        :param sequence: Sequence to scan results for
        :type sequence: Sequence

        :returns: Results for the tracker and sequence
        :rtype: Results"""
        raise NotImplementedError

    def results(self, tracker: "Tracker", sequence: "Sequence") -> "Results":
        """Get results for a given tracker and sequence.

        :param tracker: Tracker to get results for
        :type tracker: Tracker
        :param sequence: Sequence to get results for
        :type sequence: Sequence

        :returns: Results for the tracker and sequence
        :rtype: Results"""
        if tracker.storage is not None:
            return tracker.storage.results(tracker, self, sequence)
        if not hasattr(self, "_storage"):
            from vot.workspace import WorkspaceException
            raise WorkspaceException("Experiment has no storage")
        
        return self._storage.results(tracker, self, sequence)

    def log(self, identifier: str):
        """Get a log file for the experiment.

        :param identifier: Identifier of the log
        :type identifier: str

        :returns: Path to the log file
        :rtype: str"""
        if not hasattr(self, "_storage"):
            import os
            # Return a devnull file if the experiment has no storage
            return open(os.devnull, 'w') 
        
        return self._storage.substorage("logs").write("{}_{:%Y-%m-%dT%H-%M-%S.%f%z}.log".format(identifier, datetime.now()))

    def transform(self, sequences):
        """Transform a list of sequences using the experiment transformers.

        :param sequences: List of sequences to transform
        :type sequences: typing.List[Sequence]

        :returns: List of transformed sequences. The number of sequences may be larger than the input as some transformers may split sequences.
        :rtype: typing.List[Sequence]"""
        from vot.dataset import Sequence
        from vot.experiment.transformer import SingleObject
        from vot import get_logger
        
        if isinstance(sequences, Sequence):
            sequences = [sequences]
        
        transformers = list(self.transformers)

        if not self._multiobject:
            get_logger().debug("Adding single object transformer since experiment is not multi-object")
            transformers.insert(0, SingleObject(cache=None))

        # Process sequences one transformer at the time. The number of sequences may grow
        for transformer in transformers:
            transformed = []
            for sequence in sequences:
                get_logger().debug("Transforming sequence {} with transformer {}.{}".format(sequence.identifier, transformer.__class__.__module__, transformer.__class__.__name__))
                transformed.extend(transformer(sequence))
            sequences = transformed

        if self.ignore_special:
            sequences = [IgnoreSpecialObjects(sequence) for sequence in sequences]

        return sequences

from .multirun import UnsupervisedExperiment, SupervisedExperiment
from .multistart import MultiStartExperiment

def run_experiment(experiment: Experiment, tracker: "Tracker", sequences: typing.List["Sequence"], force: bool = False, persist: bool = False):
    """A helper function that performs a given experiment with a given tracker on a list
    of sequences.

    :param experiment: The experiment object
    :type experiment: Experiment
    :param tracker: The tracker object
    :type tracker: Tracker
    :param sequences: List of sequences.
    :type sequences: typing.List[Sequence]
    :param force: Ignore the cached results, rerun all the experiments. Defaults to False.
    :type force: bool, optional
    :param persist: Continue runing even if exceptions were raised. Defaults to False.
    :type persist: bool, optional

    :raises TrackerException: If the experiment is interrupted"""

    class EvaluationProgress(object):
        """A helper class that wraps a progress bar and updates it based on the number
        of finished sequences."""

        def __init__(self, description, total):
            """Initialize the progress bar.

            :param description: Description of the progress bar
            :type description: str
            :param total: Total number of sequences
            :type total: int

            :raises ValueError: If the total number of sequences is not positive"""
            self.bar = Progress(description, total)
            self._finished = 0

        def __call__(self, progress):
            """Update the progress bar. The progress is a number between 0 and 1.

            :param progress: Progress of the current sequence
            :type progress: float

            :raises ValueError: If the progress is not between 0 and 1"""
            self.bar.absolute(self._finished + min(1, max(0, progress)))

        def push(self):
            """Push the progress bar."""
            self._finished = self._finished + 1
            self.bar.absolute(self._finished)

        def close(self):
            """Close the progress bar."""
            self.bar.close()

    from vot import get_logger

    logger = get_logger()

    transformed = []
    for sequence in sequences:
        transformed.extend(experiment.transform(sequence))
    sequences = transformed

    progress = EvaluationProgress("{}/{}".format(tracker.identifier, experiment.identifier), len(sequences))
    for sequence in sequences:
        try:
            experiment.execute(tracker, sequence, force=force, callback=progress)
        except TrackerException as te:
            logger.error("Tracker %s encountered an error at sequence %s: %s", te.tracker.identifier, sequence.name, te)
            logger.debug(te, exc_info=True)
            if not te.log is None:
                with experiment.log(te.tracker.identifier) as flog:
                    flog.write(te.log)
                    logger.error("Tracker output written to file: %s", flog.name)
            if not persist:
                raise TrackerException("Experiment interrupted", te, tracker=tracker)
        progress.push()

    progress.close()
    
experiment_registry = Registry("experiment")
transformer_registry = Registry("transformer")
