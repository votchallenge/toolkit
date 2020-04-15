
import os
import re
import configparser
import logging
from typing import Tuple

from abc import abstractmethod, ABC

import yaml

from vot import VOTException
from vot.dataset import Frame
from vot.region import Region

logger = logging.getLogger("vot")

class TrackerException(VOTException):
    def __init__(self, *args, tracker, tracker_log=None):
        super().__init__(*args)
        self._tracker_log = tracker_log
        self._tracker = tracker

    @property
    def log(self):
        return self._tracker_log

    @property
    def tracker(self):
        return self._tracker

class TrackerTimeoutException(TrackerException):
    pass

VALID_IDENTIFIER = re.compile("^[a-zA-Z0-9-_]+$")

def is_valid_identifier(identifier):
    return not VALID_IDENTIFIER.match(identifier) is None

_runtime_protocols = {}

def load_trackers(directories, root=os.getcwd()):

    trackers = dict()

    logger = logging.getLogger("vot")

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

        logger.info("Scanning registry %s", registry)

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

                trackers[k] = Tracker(_identifier=k, _registry=registry, **v)

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

                trackers[section] = Tracker(_identifier=section, _registry=registry, **config[section])
    return trackers

def collect_envvars(**kwargs):
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

def collect_arguments(**kwargs):
    arguments = dict()
    other = dict()

    if "env" in kwargs:
        if isinstance(kwargs["arguments"], dict):
            arguments.update(kwargs["arguments"])
        del kwargs["arguments"]

    for name, value in kwargs.items():
        if name.startswith("arg_") and len(name) > 4:
            arguments[name[4:].lower()] = value
        else:
            other[name] = value
    
    return arguments, other

class Tracker(object):

    def __init__(self, _identifier, _registry, command, protocol=None, label=None, **kwargs):
        self._identifier = _identifier
        self._registry = _registry
        self._command = command
        self._protocol = protocol
        self._label = label
        self._envvars, args = collect_envvars(**kwargs)
        self._arguments, self._args = collect_arguments(**args)

    def runtime(self, log=False) -> "TrackerRuntime":
        if not self._protocol:
            raise TrackerException("Tracker does not have an attached executable", tracker=self)

        if not self._protocol in _runtime_protocols:
            raise TrackerException("Runtime protocol '{}' not available".format(self._protocol), tracker=self)

        return _runtime_protocols[self._protocol](self, self._command, log=log, envvars=self._envvars, arguments=self._arguments, **self._args)

    @property
    def registry(self):
        return self._registry

    @property
    def identifier(self):
        return self._identifier

    @property
    def label(self):
        return self._label

    @property
    def protocol(self):
        return self._protocol

class TrackerRuntime(ABC):

    def __init__(self, tracker: Tracker):
        self._tracker = tracker

    @property
    def tracker(self) -> Tracker:
        return self._tracker

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def restart(self):
        pass

    @abstractmethod
    def initialize(self, frame: Frame, region: Region, properties: dict = None) -> Tuple[Region, dict, float]:
        pass

    @abstractmethod
    def update(self, frame: Frame, properties: dict = None) -> Tuple[Region, dict, float]:
        pass

class RealtimeTrackerRuntime(TrackerRuntime):

    def __init__(self, runtime: TrackerRuntime, grace:int = 1, interval:float = 0.1):
        super().__init__(runtime.tracker)
        self._runtime = runtime
        self._grace = grace
        self._interval = interval
        self._countdown = 0
        self._time = 0
        self._out = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def stop(self):
        self._runtime.stop()
        self._time = 0
        self._out = None

    def restart(self):
        self._runtime.restart()
        self._time = 0
        self._out = None

    def initialize(self, frame: Frame, region: Region, properties: dict = None) -> Tuple[Region, dict, float]:
        self._countdown = self._grace
        self._out = None

        out, prop, time = self._runtime.initialize(frame, region, properties)

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


    def update(self, frame: Frame, properties: dict = None) -> Tuple[Region, dict, float]:

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

    def __init__(self, runtime: TrackerRuntime, **kwargs):
        super().__init__(runtime.tracker)
        self._runtime = runtime
        self._properties = {k : str(v) for k, v in kwargs.items()}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def stop(self):
        self._runtime.stop()

    def restart(self):
        self._runtime.restart()

    def initialize(self, frame: Frame, region: Region, properties: dict = None) -> Tuple[Region, dict, float]:
        
        if not properties is None:
            tproperties = dict(properties)
        else:
            tproperties = dict()

        tproperties.update(self._properties)

        return self._runtime.initialize(frame, region, tproperties)


    def update(self, frame: Frame, properties: dict = None) -> Tuple[Region, dict, float]:
        return self._runtime.update(frame, properties)


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