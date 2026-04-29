"""This module contains the base classes for trackers and the registry of known
trackers."""

import os
import re
import configparser
import copy
from numbers import Real
from typing import Tuple, List, Union, Dict
from collections import OrderedDict
from abc import abstractmethod, ABC

import yaml

from vot import ToolkitException
from vot.dataset import Frame
from vot.utilities import to_string
from vot.region import Region, Special

class TrackerException(ToolkitException):
    """Base class for all tracker related exceptions."""

    def __init__(self, *args, tracker, tracker_log=None):
        """Initialize the exception.

        :param tracker: Tracker that caused the exception.
        :type tracker: Tracker
        :param tracker_log: Optional log message. Defaults to None.
        :type tracker_log: str, optional
        """
        super().__init__(*args)
        self._tracker_log = tracker_log
        self._tracker = tracker

    @property
    def log(self) -> str:
        """Returns the log message of the tracker.

        :returns: Log message of the tracker.
        :rtype: sts"""
        return self._tracker_log

    @property
    def tracker(self):
        """Returns the tracker that caused the exception."""
        return self._tracker

class TrackerTimeoutException(TrackerException):
    """Exception raised when the tracker communication times out."""
    pass

VALID_IDENTIFIER = re.compile("^[a-zA-Z0-9-_]+$")

VALID_REFERENCE = re.compile("^([a-zA-Z0-9-_]+)(@[a-zA-Z0-9-_]*)?$")

def is_valid_identifier(identifier):
    """Checks if the identifier is valid.

    :param identifier: The identifier to check.
    :type identifier: str

    :returns: True if the identifier is valid, False otherwise.
    :rtype: bool"""
    return not VALID_IDENTIFIER.match(identifier) is None

def is_valid_reference(reference):
    """Checks if the reference is valid.

    :param reference: The reference to check.
    :type reference: str

    :returns: True if the reference is valid, False otherwise.
    :rtype: bool"""
    return not VALID_REFERENCE.match(reference) is None

def parse_reference(reference):
    """Parses the reference into identifier and version.

    :param reference: The reference to parse.
    :type reference: str

    :returns: A tuple containing the identifier and the version.
    :rtype: tuple
    :raises ValueError: If the reference is not valid."""
    matches = VALID_REFERENCE.match(reference)
    if not matches:
        return None, None
    return matches.group(1), matches.group(2)[1:] if not matches.group(2) is None else None

_runtime_protocols = {}

def register_runtime_protocol(protocol, constructor):
    """Registers a runtime protocol with the given constructor.

    :param protocol: The name of the protocol.
    :type protocol: str
    :param constructor: The constructor for the runtime protocol.
    :type constructor: callable
    """
    if protocol in _runtime_protocols:
        raise ValueError("Runtime protocol '{}' is already registered".format(protocol))
    
    _runtime_protocols[protocol] = constructor

class Registry(object):
    """Repository of known trackers.

    Trackers are loaded from a manifest files in one or more directories.
    """

    def __init__(self, directories, root=os.getcwd()):
        """Initialize the registry.

        :param directories: List of directories to scan for trackers.
        :type directories: list
        :param root: The root directory of the workspace. Defaults to os.getcwd().
        :type root: str, optional
        """
        from vot import get_logger
        
        logger = get_logger()
        
        trackers = dict()
        registries = []

        for directory in directories:
            if not os.path.isabs(directory):
                directory = os.path.normpath(os.path.abspath(os.path.join(root, directory)))

            if os.path.isdir(directory):
                registries.append(os.path.join(directory, "trackers.yaml"))
                registries.append(os.path.join(directory, "trackers.ini"))

            if os.path.isfile(directory):
                registries.append(directory)

        for registry in list(dict.fromkeys(registries)):
            if not os.path.isfile(registry):
                continue

            logger.debug("Scanning registry %s", registry)

            extension = os.path.splitext(registry)[1].lower()

            if extension == ".yaml":
                with open(registry, 'r') as fp:
                    metadata = yaml.load(fp, Loader=yaml.BaseLoader)
                for k, v in metadata.items():
                    if not is_valid_identifier(k):
                        logger.warning("Invalid tracker identifier %s in %s", k, registry)
                        continue
                    if k in trackers:
                        logger.warning("Duplicate tracker identifier %s in %s", k, registry)
                        continue

                    trackers[k] = Tracker(_identifier=k, _source=registry, **v)

            if extension == ".ini":
                config = configparser.ConfigParser()
                config.read(registry)
                for section in config.sections():
                    if not is_valid_identifier(section):
                        logger.warning("Invalid identifier %s in %s", section, registry)
                        continue
                    if section in trackers:
                        logger.warning("Duplicate tracker identifier %s in %s", section, registry)
                        continue

                    trackers[section] = Tracker(_identifier=section, _source=registry, **config[section])

        self._trackers = OrderedDict(sorted(trackers.items(), key=lambda t: t[0]))
        logger.debug("Found %d trackers", len(self._trackers))

    def __getitem__(self, reference) -> "Tracker":
        """Returns the tracker for the given reference."""

        return self.resolve(reference, skip_unknown=False, resolve_plural=False)[0]

    def __contains__(self, reference) -> bool:
        """Checks if the tracker is registered."""
        identifier, _ = parse_reference(reference)
        return identifier in self._trackers

    def __iter__(self):
        """Returns an iterator over the trackers."""
        return iter(self._trackers.values())

    def __len__(self):
        """Returns the number of trackers."""
        return len(self._trackers)

    def resolve(self, *references, storage=None, skip_unknown=True, resolve_plural=True):
        """Resolves the references to trackers.

        :param storage: Storage to use for resolving references. Defaults to None.
        :type storage: _type_, optional
        :param skip_unknown: Skip unknown trackers. Defaults to True.
        :type skip_unknown: bool, optional
        :param resolve_plural: Resolve plural references. Defaults to True.
        :type resolve_plural: bool, optional

        :raises ToolkitException: When a reference cannot be resolved.
        :returns: Resolved trackers.
        :rtype: list""" 

        trackers = []

        for reference in references:
            
            if resolve_plural and reference.startswith("#"):
                tag = reference[1:]
                if not is_valid_identifier(tag):
                    continue
                for tracker in self._trackers.values():
                    if tracker.tagged(tag):
                        trackers.extend(self._find_versions(tracker.identifier, storage))
                continue

            identifier, version = parse_reference(reference)

            if not identifier in self._trackers:
                if not skip_unknown:
                    raise ToolkitException("Unable to resolve tracker reference: {}".format(reference))
                else:
                    continue

            base = self._trackers[identifier]

            if version == "":
                trackers.extend(self._find_versions(identifier, storage))
            else:
                trackers.append(base.reversion(version))

        return trackers

    def _find_versions(self, identifier: str, storage: "Storage"):
        """Finds all versions of the tracker in the storage.

        :param identifier: The identifier of the tracker.
        :type identifier: str
        :param storage: The storage to use for finding the versions.
        :type storage: Storage

        :returns: List of trackers.
        :rtype: list"""

        trackers = []

        if storage is None:
            return trackers

        for reference in storage.folders():
            if reference.startswith(identifier + "@") or reference == identifier:
                identifier, version = parse_reference(reference)
                base = self._trackers[identifier]
                trackers.append(base.reversion(version))

        return trackers

    def references(self):
        """Returns a list of all tracker references.

        :returns: List of tracker references.
        :rtype: list"""
        return [t.reference for t in self._trackers.values()]

    def identifiers(self):
        """Returns a list of all tracker identifiers.

        :returns: List of tracker identifiers.
        :rtype: list"""
        return [t.identifier for t in self._trackers.values()]

class Tracker(object):
    """Tracker definition class."""

    @staticmethod
    def _collect_envvars(**kwargs):
        """Collects environment variables from the keyword arguments.

        :param **kwargs: Keyword arguments.

        :returns: Tuple of environment variables and other keyword arguments.
        :rtype: tuple"""
        envvars = dict()
        other = dict()

        if "env" in kwargs:
            if isinstance(kwargs["env"], dict):
                envvars.update({k: os.path.expandvars(v) for k, v in kwargs["env"].items()})
            del kwargs["env"]

        for name, value in kwargs.items():
            if name.startswith("env_") and len(name) > 4:
                envvars[name[4:].upper()] = os.path.expandvars(value)
            else:
                other[name] = value

        return envvars, other

    @staticmethod
    def _collect_arguments(**kwargs):
        """Collects arguments from the keyword arguments.

        :param **kwargs: Keyword arguments.

        :returns: Tuple of arguments and other keyword arguments.
        :rtype: tuple"""
        arguments = dict()
        other = dict()

        if "arguments" in kwargs:
            if isinstance(kwargs["arguments"], dict):
                arguments.update(kwargs["arguments"])
            del kwargs["arguments"]

        for name, value in kwargs.items():
            if name.startswith("arg_") and len(name) > 4:
                arguments[name[4:].lower()] = value
            else:
                other[name] = value

        return arguments, other

    @staticmethod
    def _collect_metadata(**kwargs):
        """Collects metadata from the keyword arguments.

        :param **kwargs: Keyword arguments.
            
        :returns: Tuple of metadata and other keyword arguments.
        :rtype: tuple
        Examples:
            >>> Tracker._collect_metadata(meta_author="John Doe", meta_year=2018)
            ({'author': 'John Doe', 'year': 2018}, {})
        """
        metadata = dict()
        other = dict()

        if "metadata" in kwargs:
            if isinstance(kwargs["metadata"], dict):
                metadata.update(kwargs["metadata"])
            del kwargs["arguments"]

        for name, value in kwargs.items():
            if name.startswith("meta_") and len(name) > 5:
                metadata[name[5:].lower()] = value
            else:
                other[name] = value

        return metadata, other

    def __init__(self, _identifier, _source, command, protocol=None, label=None, version=None, tags=None, storage=None, **kwargs):
        """Initializes the tracker definition.

        :param _identifier: The identifier of the tracker.
        :type _identifier: str
        :param _source: The source of the tracker.
        :type _source: str
        :param command: The command to execute.
        :type command: str
        :param protocol: The protocol of the tracker. Defaults to None.
        :type protocol: str, optional
        :param label: The label of the tracker. Defaults to None.
        :type label: str, optional
        :param version: The version of the tracker. Defaults to None.
        :type version: str, optional
        :param tags: The tags of the tracker. Defaults to None.
        :type tags: str, optional
        :param storage: The storage of the tracker. Defaults to None.
        :type storage: str, optional
        :param **kwargs: Additional keyword arguments.

        :raises ValueError: When the identifier is not valid."""
        from vot.workspace import LocalStorage
        self._identifier = _identifier
        self._source = _source
        self._command = command
        self._protocol = protocol
        self._storage = LocalStorage(storage) if storage is not None else None
        self._label = label if label is not None else _identifier
        self._version = to_string(version) if not version is None else None
        self._envvars, args = Tracker._collect_envvars(**kwargs)
        self._metadata, args = Tracker._collect_metadata(**args)
        self._arguments, self._args = Tracker._collect_arguments(**args)

        if tags is None:
            self._tags = []
        elif isinstance(tags, str):
            self._tags = tags.split(",")
        self._tags = [tag.strip() for tag in self._tags]
        self._tags = [tag for tag in self._tags if is_valid_identifier(tag)]
   
        if not self._version is None and not is_valid_identifier(self._version):
            raise TrackerException("Illegal version format", tracker=self)

    def reversion(self, version=None) -> "Tracker":
        """Creates a new tracker instance for specified version.

            version {[type]} -- New version (default: {None})

        :returns: Tracker -- [description]"""
        if self.version == version or version is None:
            return self
        tracker = copy.copy(self)
        tracker._version = version
        return tracker

    def runtime(self, log=False) -> "TrackerRuntime":
        """Creates a new runtime instance for this tracker instance."""
        if not self._command:
            raise TrackerException("Tracker does not have an attached executable", tracker=self)

        if self._protocol is None:
            raise TrackerException("Tracker does not have an attached protocol and can not be executed", tracker=self)

        if not self._protocol in _runtime_protocols:
            raise TrackerException("Runtime protocol '{}' not available".format(self._protocol), tracker=self)

        return _runtime_protocols[self._protocol](self, self._command, log=log, envvars=self._envvars, arguments=self._arguments, **self._args)

    def __eq__(self, other):
        """Checks if two trackers are equal.

        :param other: The other tracker.
        :type other: Tracker

        :returns: True if the trackers are equal, False otherwise.
        :rtype: bool"""
        if other is None or not isinstance(other, Tracker):
            return False

        return self.reference == other.identifier

    def __hash__(self):
        """Returns the hash of the tracker."""
        return hash(self.reference)

    def __repr__(self):
        """Returns the string representation of the tracker."""
        return self.reference

    @property
    def source(self):
        """Returns the source of the tracker."""
        return self._source

    @property
    def storage(self) -> "Storage":
        """Returns the storage of the tracker results."""
        return self._storage

    @property
    def identifier(self) -> str:
        """Returns the identifier of the tracker."""
        return self._identifier

    @property
    def label(self):
        """Returns the label of the tracker. If the version is specified, the label will
        contain the version as well.

        :returns: Label of the tracker.
        :rtype: str"""
        if self._version is None:
            return self._label
        else:
            return self._label + " (" + self._version + ")"

    @property
    def version(self) -> str:
        """Returns the version of the tracker. If the version is not specified, None is
        returned.

        :returns: Version of the tracker.
        :rtype: str"""
        return self._version

    @property
    def reference(self) -> str:
        """Returns the reference of the tracker. If the version is specified, the
        reference will contain the version as well.

        :returns: Reference of the tracker.
        :rtype: str"""
        if self._version is None:
            return self._identifier
        else:
            return self._identifier + "@" + self._version

    @property
    def protocol(self) -> str:
        """Returns the communication protocol used by this tracker.

        :returns: Communication protocol
        :rtype: str"""
        return self._protocol

    def describe(self):
        """Returns a dictionary containing the tracker description.

        :returns: Dictionary containing the tracker description.
        :rtype: dict"""
        data = dict(command=self._command, label=self.label, protocol=self.protocol, arguments=self._arguments, env=self._envvars)
        data.update(self._args)
        return data

    def metadata(self, key):
        """Returns the metadata value for specified key."""
        if not key in self._metadata:
            return None
        return self._metadata[key]

    def tagged(self, tag):
        """Returns true if the tracker is tagged with specified tag.

        :param tag: The tag to check.
        :type tag: str

        :returns: True if the tracker is tagged with specified tag, False otherwise.
        :rtype: bool"""
        for t in self._tags:
            if t == tag:
                return True
        return False

class ObjectStatus(tuple):
    """Tuple-like immutable container for an object state."""

    __slots__ = ()
    _fields = ("region", "properties")

    def __new__(cls, region, properties):
        if not isinstance(region, Region):
            raise TypeError("ObjectStatus.region must be a Region")
        if not isinstance(properties, dict):
            raise TypeError("ObjectStatus.properties must be a dict")
        return tuple.__new__(cls, (region, properties))

    @property
    def region(self):
        return self[0]

    @property
    def properties(self):
        return self[1]

    @classmethod
    def _make(cls, iterable):
        region, properties = iterable
        return cls(region, properties)

    def _asdict(self):
        return OrderedDict(zip(self._fields, self))

    def _replace(self, **kwargs):
        return self.__class__(
            kwargs.get("region", self.region),
            kwargs.get("properties", self.properties)
        )


class ObjectQuery(tuple):
    """Tuple-like immutable container for an object query."""

    __slots__ = ()
    _fields = ("state", "properties", "offset")

    def __new__(cls, state, properties, offset):
        if not isinstance(state, Region):
            raise TypeError("ObjectQuery.state must be a Region")
        if not isinstance(properties, dict):
            raise TypeError("ObjectQuery.properties must be a dict")
        if not isinstance(offset, int):
            raise TypeError("ObjectQuery.offset must be an int")
        return tuple.__new__(cls, (state, properties, offset))

    @property
    def state(self):
        return self[0]

    @property
    def properties(self):
        return self[1]

    @property
    def offset(self):
        return self[2]

    @classmethod
    def _make(cls, iterable):
        state, properties, offset = iterable
        return cls(state, properties, offset)

    def _asdict(self):
        return OrderedDict(zip(self._fields, self))

    def _replace(self, **kwargs):
        return self.__class__(
            kwargs.get("state", self.state),
            kwargs.get("properties", self.properties),
            kwargs.get("offset", self.offset)
        )


class FrameResult(tuple):
    """Tuple-like immutable container for per-frame tracker output."""

    __slots__ = ()
    _fields = ("objects", "time")

    def __new__(cls, objects, time):
        if not isinstance(time, Real):
            raise TypeError("FrameResult.time must be a real number")
        return tuple.__new__(cls, (objects, float(time)))

    @property
    def objects(self):
        return self[0]

    @property
    def time(self):
        return self[1]

    @classmethod
    def _make(cls, iterable):
        objects, time = iterable
        return cls(objects, time)

    def _asdict(self):
        return OrderedDict(zip(self._fields, self))

    def _replace(self, **kwargs):
        return self.__class__(
            kwargs.get("objects", self.objects),
            kwargs.get("time", self.time)
        )


class RunResult(tuple):
    """Tuple-like immutable container for full-run tracker output."""

    __slots__ = ()
    _fields = ("objects", "times")

    def __new__(cls, objects, times):
        if not isinstance(objects, list):
            raise TypeError("RunResult.objects must be a list")
        if not isinstance(times, list):
            raise TypeError("RunResult.times must be a list")
        if not all(isinstance(t, Real) for t in times):
            raise TypeError("RunResult.times must contain only real numbers")
        return tuple.__new__(cls, (objects, [float(t) for t in times]))

    @property
    def objects(self):
        return self[0]

    @property
    def times(self):
        return self[1]

    @classmethod
    def _make(cls, iterable):
        objects, times = iterable
        return cls(objects, times)

    def _asdict(self):
        return OrderedDict(zip(self._fields, self))

    def _replace(self, **kwargs):
        return self.__class__(
            kwargs.get("objects", self.objects),
            kwargs.get("times", self.times)
        )


RunQueries = List[ObjectQuery]

FrameObjects = Union[List[ObjectStatus], ObjectStatus, None]


class TrackerRuntime(ABC):
    """Base class for tracker runtime implementations.

    Tracker runtime is responsible for running the tracker executable and communicating
    with it.
    """

    def __init__(self, tracker: Tracker):
        """Creates a new tracker runtime instance.

        :param tracker: The tracker instance.
        :type tracker: Tracker
        """
        self._tracker = tracker

    def run(self, frames: List[Frame], queries: RunQueries) -> RunResult: 
        """
        Runs the tracker on the specified frames and queries. 
        Returns a dictionary containing the objects for each query.
        
        Args:
            frames (List[Frame]): The frames to run the tracker on.
            queries (Dict[str, ObjectQuery]): The queries to run the tracker on.

        Returns:
            RunResult: A run result containing the objects and times for each query.
        """
        
        raise NotImplementedError("TrackerRuntime.run() is not implemented. Please implement the run() method in the tracker runtime implementation.")

    @property
    def tracker(self) -> Tracker:
        """Returns the tracker instance associated with this runtime."""
        return self._tracker

    def __enter__(self):
        """Starts the tracker runtime."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stops the tracker runtime."""
        self.stop()

    @property
    def multiobject(self):
        """Returns True if the tracker supports multiple objects, False otherwise."""
        return False

    @abstractmethod
    def stop(self):
        """Stops the tracker runtime."""
        raise NotImplementedError


class OnlineTrackerRuntime(TrackerRuntime):
    """Base class for online tracker runtime implementations.

    Tracker runtime is responsible for running the tracker executable and communicating
    with it.
    """

    def __init__(self, tracker: Tracker):
        """Creates a new tracker runtime instance.

        :param tracker: The tracker instance.
        :type tracker: Tracker
        """
        super().__init__(tracker)
        self._tracker = tracker

    @property
    def tracker(self) -> Tracker:
        """Returns the tracker instance associated with this runtime."""
        return self._tracker

    def __enter__(self):
        """Starts the tracker runtime."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stops the tracker runtime."""
        self.stop()

    @property
    def multiobject(self):
        """Returns True if the tracker supports multiple objects, False otherwise."""
        return False

    @abstractmethod
    def stop(self):
        """Stops the tracker runtime."""
        raise NotImplementedError

    @abstractmethod
    def restart(self):
        """Restarts the tracker runtime, usually stars a new process."""
        raise NotImplementedError

    @abstractmethod
    def initialize(self, frame: Frame, new: FrameObjects = None) -> Tuple[FrameObjects, float]:
        """Initializes the tracker runtime with specified frame and objects. Returns the initial objects and the time it took to initialize the tracker.
        
        :param frame: The frame to initialize the tracker with.
        :param new: The objects to initialize the tracker with. 

        :returns: Tuple[Objects, float] -- The initial objects and the time it took to initialize the tracker."""
        raise NotImplementedError

    @abstractmethod
    def update(self, frame: Frame, new: FrameObjects = None) -> Tuple[FrameObjects, float]:
        """Updates the tracker runtime with specified frame and objects. Returns the
        updated objects and the time it took to update the tracker.

        :param frame: The frame to update the tracker with.
        :param new: The objects to update the tracker with.
        
        :returns: Tuple[Objects, float] -- The updated objects and the time it took to update the tracker.
        """
        raise NotImplementedError

    def run(self, frames: List[Frame], queries: RunQueries) -> RunResult:
        """Runs the tracker on the given frames and queries. 
        Returns the tracker output as a RunStatus namedtuple.
        The online tracker runtime uses the interface defined by 
        the initialize and update methods to run the tracker 
        on the given frames and queries.

        :param frames: The frames to run the tracker on.
        :param queries: The queries to run the tracker on.
        """
        
        # Order the queries by offset and id
        
        statuses = []
        # Initialize statuses with empty lists for each query
        for i in range(len(queries)):
            statuses.append([])
        
        times = []
        
        for i, frame in enumerate(frames):
            # Filter out objects appearing in the current frame
            new = [ObjectStatus(queries[j].state, queries[j].properties) for j in range(len(queries)) if queries[j].offset == i]
        
            if i == 0:
                status, time = self.initialize(frames[0], new)
            else:
                status, time = self.update(frame, new)
        
            times.append(time)
        
            for j in range(len(queries)):
                if queries[j].offset <= i:
                    statuses[j].append(status[j])
                else:
                    statuses[j].append(ObjectStatus(Special(Trajectory.UNKNOWN), {})) 
                   
        return RunResult(statuses, times)

class RealtimeTrackerRuntime(TrackerRuntime):
    """Base class for realtime tracker runtime implementations.

    Realtime tracker runtime is responsible for running the tracker executable and
    communicating with it while simulating given real-time constraints.
    """

    def __init__(self, runtime: TrackerRuntime, grace: int = 1, interval: float = 0.1):
        """Initializes the realtime tracker runtime with specified tracker runtime,
        grace period and update interval.

            runtime (OnlineTrackerRuntime) -- The tracker runtime to wrap.
            grace (int) -- The grace period in seconds. The tracker will be updated at least once during the grace period. (default: {1})
            interval (float) -- The update interval in seconds. (default: {0.1})
        """
        if not isinstance(runtime, OnlineTrackerRuntime):
            raise ValueError("Runtime does not support online communication")
        
        super().__init__(runtime.tracker)
        self._runtime = runtime
        self._grace = grace
        self._interval = interval
        self._countdown = 0
        self._time = 0
        self._status = None

    @property
    def multiobject(self):
        """Returns True if the tracker supports multiple objects, False otherwise."""
        return self._runtime.multiobject

    def stop(self):
        """Stops the tracker runtime."""
        self._runtime.stop()
        self._time = 0
        self._status = None

    def restart(self):
        """Restarts the tracker runtime, usually stars a new process."""
        self._runtime.restart()
        self._time = 0
        self._status = None

    def initialize(self, frame: Frame, new: FrameObjects = None) -> Tuple[FrameObjects, float]:
        """Initializes the tracker runtime with specified frame and objects. Returns the
        initial objects and the time it took to initialize the tracker.

        :param frame: The frame to initialize the tracker with.
        :param new: The objects to initialize the tracker with.
        
        :returns: Tuple[Objects, float] -- The initial objects and the time it took to initialize the tracker."""
        self._countdown = self._grace
        self._status = None

        status, time = self._runtime.initialize(frame, new)

        if time > self._interval:
            if self._countdown > 0:
                self._countdown = self._countdown - 1
                self._time = 0
            else:
                self._time = time - self._interval
                self._status = status
        else:
            self._time = 0

        return status, time


    def update(self, frame: Frame, new: FrameObjects = None) -> Tuple[FrameObjects, float]:
        """Updates the tracker runtime with specified frame and objects. Returns the
        updated objects and the time it took to update the tracker.

        Note that adding new objects is not supported in realtime tracker runtime, as a frame may be skipped
        if the tracker fails to update within the specified interval.

        :param frame: The frame to update the tracker with.
        :param new: The objects to update the tracker with. Setting new objects is not supported in realtime tracker and will raise an assertion error.

        :returns: Tuple[Objects, float] -- The updated objects and the time it took to update the tracker."""

        assert new is None or len(new) == 0, "Adding new objects is not supported in realtime tracker runtime"

        if self._time > self._interval:
            self._time = self._time - self._interval
            return self._status, 0
        else:
            self._status = None
            self._time = 0

        status, time = self._runtime.update(frame, None)

        if time > self._interval:
            if self._countdown > 0:
                self._countdown = self._countdown - 1
                self._time = 0
            else:
                self._time = time - self._interval
                self._status = status

        return status, time

try:
    import vot.tracker.trax
except OSError:
    pass
except ImportError:
    from vot import get_logger
    get_logger().warning("Unable to import support for TraX protocol")

from vot.tracker.folder import TrackerFolderRuntime
from vot.tracker.python import PythonRuntime

from vot.tracker.results import Trajectory, Results