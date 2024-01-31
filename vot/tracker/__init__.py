""" This module contains the base classes for trackers and the registry of known trackers. """

import os
import re
import configparser
import logging
import copy
from typing import Tuple, List, Union
from collections import OrderedDict, namedtuple
from abc import abstractmethod, ABC

import yaml

from vot import ToolkitException
from vot.dataset import Frame
from vot.region import Region
from vot.utilities import to_string

logger = logging.getLogger("vot")

class TrackerException(ToolkitException):
    """ Base class for all tracker related exceptions."""

    def __init__(self, *args, tracker, tracker_log=None):
        """ Initialize the exception.

        Args:
            tracker (Tracker): Tracker that caused the exception.
            tracker_log (str, optional): Optional log message. Defaults to None.
        """
        super().__init__(*args)
        self._tracker_log = tracker_log
        self._tracker = tracker

    @property
    def log(self) -> str:
        """ Returns the log message of the tracker.

        Returns:
            sts: Log message of the tracker.
        """
        return self._tracker_log

    @property
    def tracker(self):
        """ Returns the tracker that caused the exception."""
        return self._tracker

class TrackerTimeoutException(TrackerException):
    """ Exception raised when the tracker communication times out."""
    pass

VALID_IDENTIFIER = re.compile("^[a-zA-Z0-9-_]+$")

VALID_REFERENCE = re.compile("^([a-zA-Z0-9-_]+)(@[a-zA-Z0-9-_]*)?$")

def is_valid_identifier(identifier):
    """Checks if the identifier is valid.
    
    Args:
        identifier (str): The identifier to check.
        
    Returns:
        bool: True if the identifier is valid, False otherwise.
    """
    return not VALID_IDENTIFIER.match(identifier) is None

def is_valid_reference(reference):
    """Checks if the reference is valid.
    
    Args:
        reference (str): The reference to check.
        
    Returns:
        bool: True if the reference is valid, False otherwise.
    """
    return not VALID_REFERENCE.match(reference) is None

def parse_reference(reference):
    """Parses the reference into identifier and version.
    
    Args:
        reference (str): The reference to parse.
        
    Returns:
        tuple: A tuple containing the identifier and the version.
        
    Raises:
        ValueError: If the reference is not valid.
    """
    matches = VALID_REFERENCE.match(reference)
    if not matches:
        return None, None
    return matches.group(1), matches.group(2)[1:] if not matches.group(2) is None else None

_runtime_protocols = {}

class Registry(object):
    """ Repository of known trackers. Trackers are loaded from a manifest files in one or more directories. """

    def __init__(self, directories, root=os.getcwd()):
        """ Initialize the registry.

        Args:
            directories (list): List of directories to scan for trackers.
            root (str, optional): The root directory of the workspace. Defaults to os.getcwd().
        """
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
        """ Returns the tracker for the given reference. """

        return self.resolve(reference, skip_unknown=False, resolve_plural=False)[0]

    def __contains__(self, reference) -> bool:
        """ Checks if the tracker is registered. """
        identifier, _ = parse_reference(reference)
        return identifier in self._trackers

    def __iter__(self):
        """ Returns an iterator over the trackers."""
        return iter(self._trackers.values())

    def __len__(self):
        """ Returns the number of trackers."""
        return len(self._trackers)

    def resolve(self, *references, storage=None, skip_unknown=True, resolve_plural=True):
        """ Resolves the references to trackers.

        Args:
            storage (_type_, optional): Storage to use for resolving references. Defaults to None.
            skip_unknown (bool, optional): Skip unknown trackers. Defaults to True.
            resolve_plural (bool, optional): Resolve plural references. Defaults to True.

        Raises:
            ToolkitException: When a reference cannot be resolved.

        Returns:
            list: Resolved trackers.
        """ 

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
        """ Finds all versions of the tracker in the storage. 
        
        Args:
            identifier (str): The identifier of the tracker.
            storage (Storage): The storage to use for finding the versions.
            
        Returns:
            list: List of trackers.
        """

        trackers = []

        if storage is None:
            return trackers

        for reference in storage.folders():
            if reference.startswith(identifier + "@"):
                identifier, version = parse_reference(reference)
                base = self._trackers[identifier]
                trackers.append(base.reversion(version))

        return trackers

    def references(self):
        """ Returns a list of all tracker references. 
        
        Returns:
            list: List of tracker references.
        """
        return [t.reference for t in self._trackers.values()]

    def identifiers(self):
        """ Returns a list of all tracker identifiers.
        
        Returns:
            list: List of tracker identifiers.
        """
        return [t.identifier for t in self._trackers.values()]

class Tracker(object):
    """ Tracker definition class.  """

    @staticmethod
    def _collect_envvars(**kwargs):
        """ Collects environment variables from the keyword arguments. 
         
        Args:
            **kwargs: Keyword arguments.
            
        Returns:
            tuple: Tuple of environment variables and other keyword arguments. """
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
        """ Collects arguments from the keyword arguments. 
        
        Args:
            **kwargs: Keyword arguments.
            
        Returns:
            tuple: Tuple of arguments and other keyword arguments.
        """
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
        """ Collects metadata from the keyword arguments.
        
        Args:
            **kwargs: Keyword arguments.
            
        Returns:
            tuple: Tuple of metadata and other keyword arguments.
            
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
        """ Initializes the tracker definition. 
        
        Args:
            _identifier (str): The identifier of the tracker.
            _source (str): The source of the tracker.
            command (str): The command to execute.
            protocol (str, optional): The protocol of the tracker. Defaults to None.
            label (str, optional): The label of the tracker. Defaults to None.
            version (str, optional): The version of the tracker. Defaults to None.
            tags (str, optional): The tags of the tracker. Defaults to None.
            storage (str, optional): The storage of the tracker. Defaults to None.
            **kwargs: Additional keyword arguments.
            
        Raises:
            ValueError: When the identifier is not valid.
            
        """
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
        """Creates a new tracker instance for specified version

        Keyword Arguments:
            version {[type]} -- New version (default: {None})

        Returns:
            Tracker -- [description]
        """
        if self.version == version or version is None:
            return self
        tracker = copy.copy(self)
        tracker._version = version
        return tracker

    def runtime(self, log=False) -> "TrackerRuntime":
        """Creates a new runtime instance for this tracker instance."""
        if not self._command:
            raise TrackerException("Tracker does not have an attached executable", tracker=self)

        if not self._protocol in _runtime_protocols:
            raise TrackerException("Runtime protocol '{}' not available".format(self._protocol), tracker=self)

        return _runtime_protocols[self._protocol](self, self._command, log=log, envvars=self._envvars, arguments=self._arguments, **self._args)

    def __eq__(self, other):
        """ Checks if two trackers are equal.
        
        Args:
            other (Tracker): The other tracker.
            
        Returns:
            bool: True if the trackers are equal, False otherwise.
        """
        if other is None or not isinstance(other, Tracker):
            return False

        return self.reference == other.identifier

    def __hash__(self):
        """ Returns the hash of the tracker. """
        return hash(self.reference)

    def __repr__(self):
        """ Returns the string representation of the tracker. """
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
        """Returns the label of the tracker. If the version is specified, the label will contain the version as well.
        
        Returns:
            str: Label of the tracker.
        """
        if self._version is None:
            return self._label
        else:
            return self._label + " (" + self._version + ")"

    @property
    def version(self) -> str:
        """Returns the version of the tracker. If the version is not specified, None is returned.
        
        Returns:
            str: Version of the tracker.
        """
        return self._version

    @property
    def reference(self) -> str:
        """Returns the reference of the tracker. If the version is specified, the reference will contain the version as well.
        
        Returns:
            str: Reference of the tracker.
        """
        if self._version is None:
            return self._identifier
        else:
            return self._identifier + "@" + self._version

    @property
    def protocol(self) -> str:
        """Returns the communication protocol used by this tracker.

        Returns:
            str: Communication protocol
        """
        return self._protocol

    def describe(self):
        """Returns a dictionary containing the tracker description. 
        
        Returns:
            dict: Dictionary containing the tracker description.
        """
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
        
        Args:
            tag (str): The tag to check.
            
        Returns:
            bool: True if the tracker is tagged with specified tag, False otherwise.
        """

        return tag in self._tags

ObjectStatus = namedtuple("ObjectStatus", ["region", "properties"])

Objects = Union[List[ObjectStatus], ObjectStatus]
class TrackerRuntime(ABC):
    """Base class for tracker runtime implementations. Tracker runtime is responsible for running the tracker executable and communicating with it."""

    def __init__(self, tracker: Tracker):
        """Creates a new tracker runtime instance.

        Args:
            tracker (Tracker): The tracker instance.
        """
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
        pass

    @abstractmethod
    def restart(self):
        """Restarts the tracker runtime, usually stars a new process."""
        pass

    @abstractmethod
    def initialize(self, frame: Frame, new: Objects = None, properties: dict = None) -> Tuple[Objects, float]:
        """Initializes the tracker runtime with specified frame and objects. Returns the initial objects and the time it took to initialize the tracker.
        
        Arguments:
            frame {Frame} -- The frame to initialize the tracker with.
            new {Objects} -- The objects to initialize the tracker with.
            properties {dict} -- The properties to initialize the tracker with.

        Returns:
            Tuple[Objects, float] -- The initial objects and the time it took to initialize the tracker.
        """
        pass

    @abstractmethod
    def update(self, frame: Frame, new: Objects = None, properties: dict = None) -> Tuple[Objects, float]:
        """Updates the tracker runtime with specified frame and objects. Returns the updated objects and the time it took to update the tracker.

        Arguments:
            frame {Frame} -- The frame to update the tracker with.
            new {Objects} -- The objects to update the tracker with.
            properties {dict} -- The properties to update the tracker with.

        Returns:
            Tuple[Objects, float] -- The updated objects and the time it took to update the tracker.
        """
        pass

class RealtimeTrackerRuntime(TrackerRuntime):
    """Base class for realtime tracker runtime implementations. 
    Realtime tracker runtime is responsible for running the tracker executable and communicating with it while simulating given real-time constraints."""

    def __init__(self, runtime: TrackerRuntime, grace: int = 1, interval: float = 0.1):
        """Initializes the realtime tracker runtime with specified tracker runtime, grace period and update interval.
        
        Arguments: 
            runtime {TrackerRuntime} -- The tracker runtime to wrap.
            grace {int} -- The grace period in seconds. The tracker will be updated at least once during the grace period. (default: {1})
            interval {float} -- The update interval in seconds. (default: {0.1})
            
        """
        super().__init__(runtime.tracker)
        self._runtime = runtime
        self._grace = grace
        self._interval = interval
        self._countdown = 0
        self._time = 0
        self._out = None

    @property
    def multiobject(self):
        """Returns True if the tracker supports multiple objects, False otherwise."""
        return self._runtime.multiobject

    def stop(self):
        """Stops the tracker runtime."""
        self._runtime.stop()
        self._time = 0
        self._out = None

    def restart(self):
        """Restarts the tracker runtime, usually stars a new process."""
        self._runtime.restart()
        self._time = 0
        self._out = None

    def initialize(self, frame: Frame, new: Objects = None, properties: dict = None) -> Tuple[Objects, float]:
        """Initializes the tracker runtime with specified frame and objects. Returns the initial objects and the time it took to initialize the tracker.
        
        Arguments:
            frame {Frame} -- The frame to initialize the tracker with.
            new {Objects} -- The objects to initialize the tracker with.
            properties {dict} -- The properties to initialize the tracker with.
            
        Returns:
            Tuple[Objects, float] -- The initial objects and the time it took to initialize the tracker.
        """
        self._countdown = self._grace
        self._out = None

        out, prop, time = self._runtime.initialize(frame, new, properties)

        if time > self._interval:
            if self._countdown > 0:
                self._countdown = self._countdown - 1
                self._time = 0
            else:
                self._time = time - self._interval
                self._out = out
        else:
            self._time = 0

        return out, prop, time


    def update(self, frame: Frame, _: Objects = None, properties: dict = None) -> Tuple[Objects, float]:
        """Updates the tracker runtime with specified frame and objects. Returns the updated objects and the time it took to update the tracker.
        
        Arguments:
            frame {Frame} -- The frame to update the tracker with.
            new {Objects} -- The objects to update the tracker with.
            properties {dict} -- The properties to update the tracker with.
            
        Returns:
            Tuple[Objects, float] -- The updated objects and the time it took to update the tracker.
        """

        if self._time > self._interval:
            self._time = self._time - self._interval
            return self._out, dict(), 0
        else:
            self._out = None
            self._time = 0

        out, prop, time = self._runtime.update(frame, properties)

        if time > self._interval:
            if self._countdown > 0:
                self._countdown = self._countdown - 1
                self._time = 0
            else:
                self._time = time - self._interval
                self._out = out

        return out, prop, time


class PropertyInjectorTrackerRuntime(TrackerRuntime):
    """Base class for tracker runtime implementations that inject properties into the tracker runtime."""

    def __init__(self, runtime: TrackerRuntime, **kwargs):
        """Initializes the property injector tracker runtime with specified tracker runtime and properties.

        Arguments:
            runtime {TrackerRuntime} -- The tracker runtime to wrap.
            **kwargs -- The properties to inject into the tracker runtime.
        """
        super().__init__(runtime.tracker)
        self._runtime = runtime
        self._properties = {k : str(v) for k, v in kwargs.items()}

    @property
    def multiobject(self):
        """Returns True if the tracker supports multiple objects, False otherwise."""
        return self._runtime.multiobject

    def stop(self):
        """Stops the tracker runtime."""
        self._runtime.stop()

    def restart(self):
        """Restarts the tracker runtime, usually stars a new process."""
        self._runtime.restart()

    def initialize(self, frame: Frame, new: Objects = None, properties: dict = None) -> Tuple[Objects, float]:
        """Initializes the tracker runtime with specified frame and objects. Returns the initial objects and the time it took to initialize the tracker.
        This method injects the properties into the tracker runtime.
        
        Arguments:
            frame {Frame} -- The frame to initialize the tracker with.
            new {Objects} -- The objects to initialize the tracker with.
            properties {dict} -- The properties to initialize the tracker with.
            
        Returns:
            Tuple[Objects, float] -- The initial objects and the time it took to initialize the tracker."""

        if not properties is None:
            tproperties = dict(properties)
        else:
            tproperties = dict()

        tproperties.update(self._properties)

        return self._runtime.initialize(frame, new, tproperties)


    def update(self, frame: Frame, new: Objects = None, properties: dict = None) -> Tuple[Objects, float]:
        """Updates the tracker runtime with specified frame and objects. Returns the updated objects and the time it took to update the tracker.

        Arguments:
            frame {Frame} -- The frame to update the tracker with.
            new {Objects} -- The objects to update the tracker with.
            properties {dict} -- The properties to update the tracker with.

        Returns:
            Tuple[Objects, float] -- The updated objects and the time it took to update the tracker.

        """
        return self._runtime.update(frame, new, properties)


class SingleObjectTrackerRuntime(TrackerRuntime):
    """Wrapper for tracker runtime that only support single object tracking. Used to enforce single object tracking even for multi object trackers."""

    def __init__(self, runtime: TrackerRuntime):
        """Initializes the single object tracker runtime with specified tracker runtime.
        
        Arguments:
            runtime {TrackerRuntime} -- The tracker runtime to wrap.
        """
        super().__init__(runtime.tracker)
        self._runtime = runtime

    @property
    def multiobject(self):
        """Returns False, since the tracker runtime only supports single object tracking."""
        return False

    def stop(self):
        """Stops the tracker runtime.
        
        Raises:
            TrackerException -- If the tracker runtime does not support stopping.
        """
        self._runtime.stop()

    def restart(self):
        """Restarts the tracker runtime, usually stars a new process. 

        Raises:
            TrackerException -- If the tracker runtime does not support restarting.
        """
        self._runtime.restart()

    def initialize(self, frame: Frame, new: Objects = None, properties: dict = None) -> Tuple[Objects, float]:
        """Initializes the tracker runtime with specified frame and objects. Returns the initial objects and the time it took to initialize the tracker.

        Arguments:
            frame {Frame} -- The frame to initialize the tracker with.
            new {Objects} -- The objects to initialize the tracker with.
            properties {dict} -- The properties to initialize the tracker with.
        
        Returns:
            Tuple[Objects, float] -- The initial objects and the time it took to initialize the tracker.
        """

        if isinstance(new, list) and len(new) != 1: raise TrackerException("Only supports single object tracking", tracker=self.tracker)
        status, time = self._runtime.initialize(frame, new, properties)
        if isinstance(status, list): status = status[0]
        return status, time

    def update(self, frame: Frame, new: Objects = None, properties: dict = None) -> Tuple[Objects, float]:
        """Updates the tracker runtime with specified frame and objects. Returns the updated objects and the time it took to update the tracker.
        
        Arguments:
            frame {Frame} -- The frame to update the tracker with.
            new {Objects} -- The objects to update the tracker with.
            properties {dict} -- The properties to update the tracker with.
            
        Returns:
            Tuple[Objects, float] -- The updated objects and the time it took to update the tracker.
        """

        if not new is None: raise TrackerException("Only supports single object tracking", tracker=self.tracker)
        status, time = self._runtime.update(frame, new, properties)
        if isinstance(status, list): status = status[0]
        return status, time

class MultiObjectTrackerRuntime(TrackerRuntime):
    """ This is a wrapper for tracker runtimes that do not support multi object tracking. STILL IN DEVELOPMENT!"""

    def __init__(self, runtime: TrackerRuntime):
        """Initializes the multi object tracker runtime with specified tracker runtime.

        Arguments:
            runtime {TrackerRuntime} -- The tracker runtime to wrap.
        """

        super().__init__(runtime.tracker)
        if runtime.multiobject:
            self._runtime = runtime
        else:
            self._runtime = [runtime]
            self._used = 0

    @property
    def multiobject(self):
        """Always returns True, since the tracker runtime supports multi object tracking."""
        return True

    def stop(self):
        """Stops the tracker runtime."""
        if isinstance(self._runtime, TrackerRuntime):
            self._runtime.stop()
        else:
            for r in self._runtime:
                r.stop()

    def restart(self):
        """Restarts the tracker runtime, usually stars a new process."""
        if isinstance(self._runtime, TrackerRuntime):
            self._runtime.restart()
        else:
            for r in self._runtime:
                r.restart()

    def initialize(self, frame: Frame, new: Objects = None, properties: dict = None) -> Tuple[Objects, float]:
        """Initializes the tracker runtime with specified frame and objects. Returns the initial objects and the time it took to initialize the tracker.
        Internally this method initializes the tracker runtime for each object in the objects list.
        
        Arguments:
            frame {Frame} -- The frame to initialize the tracker with.
            new {Objects} -- The objects to initialize the tracker with.
            properties {dict} -- The properties to initialize the tracker with.
            
        Returns:
            Tuple[Objects, float] -- The initial objects and the time it took to initialize the tracker.
        """

        if isinstance(self._runtime, TrackerRuntime):
            return self._runtime.initialize(frame, new, properties)
        if isinstance(new, ObjectStatus):
            new = [new]

        self._used = 0
        status = []
        for i, o in enumerate(new):
            if i >= len(self._runtime):
                self._runtime.append(self._tracker.runtime())
                self._runtime.initialize(frame, new, properties)

        if isinstance(status, list): status = status[0]
        return status

    def update(self, frame: Frame, new: Objects = None, properties: dict = None) -> Tuple[Objects, float]:
        """Updates the tracker runtime with specified frame and objects. Returns the updated objects and the time it took to update the tracker.
        Internally this method updates the tracker runtime for each object in the new objects list.

        Arguments:
            frame {Frame} -- The frame to update the tracker with.
            new {Objects} -- The objects to update the tracker with.
            properties {dict} -- The properties to update the tracker with. 

        Returns:
            Tuple[Objects, float] -- The updated objects and the time it took to update the tracker.
        """

        if not new is None: raise TrackerException("Only supports single object tracking")
        status = self._runtime.update(frame, new, properties)
        if isinstance(status, list): status = status[0]
        return status

try:

    from vot.tracker.trax import TraxTrackerRuntime, trax_matlab_adapter, trax_python_adapter, trax_octave_adapter

    _runtime_protocols["trax"] = TraxTrackerRuntime
    _runtime_protocols["traxmatlab"] = trax_matlab_adapter
    _runtime_protocols["traxpython"] = trax_python_adapter
    _runtime_protocols["traxoctave"] = trax_octave_adapter

except OSError:
    pass

except ImportError:
    logger.error("Unable to import support for TraX protocol")
    pass

from vot.tracker.results import Trajectory, Results