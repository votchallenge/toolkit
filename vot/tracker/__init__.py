
import os, re
import configparser
import yaml
import logging
from typing import Tuple

from abc import abstractmethod, ABC

from vot import VOTException
from vot.dataset import Frame
from vot.region import Region

class TrackerException(VOTException):
    pass

class TrackerTimeoutException(VOTException):
    pass

VALID_IDENTIFIER = re.compile("^[a-zA-Z0-9_]+$")

def is_valid_identifier(identifier):
    return not VALID_IDENTIFIER.match(identifier) is None

_runtime_protocols = {}

def load_trackers(directories, root=os.getcwd()):

    trackers = dict()

    logger = logging.getLogger("vot")

    for directory in directories:
        logger.info("Scanning directory %s", directory)
        if not os.path.isabs(directory):
            directory = os.path.normpath(os.path.abspath(os.path.join(root, directory)))
        if not os.path.isdir(directory):
            continue
        
        for name in [os.path.join(directory, x) for x in os.listdir(directory)]:
            if not os.path.isfile(name):
                continue
            if name.endswith(".yml") or name.endswith(".yaml"):
                with open(name, 'r') as fp:
                    metadata = yaml.load(fp, Loader=yaml.BaseLoader)
                for k, v in metadata.items():
                    if not is_valid_identifier(k):
                        raise TrackerException("Invalid identifier: {}".format(k))
                    print(k, v)
                    trackers[k] = Tracker(identifier= k, **v)

            if name.endswith(".ini"):
                config = configparser.ConfigParser()
                config.read(name)
                for section in config.sections():
                    if not is_valid_identifier(section):
                        raise TrackerException("Invalid identifier: {}".format(section))
                    trackers[section] = Tracker(identifier = section, **config[section])
    return trackers

class Tracker(object):

    def __init__(self, identifier, command, protocol=None, label=None, **kwargs):
        self._identifier = identifier
        self._command = command
        self._protocol = protocol
        self._label = label
        self._args = kwargs

    def runtime(self, log=False) -> "TrackerRuntime":
        if not self._protocol:
            raise TrackerException("Tracker does not have an attached executable")

        if not self._protocol in _runtime_protocols:
            raise TrackerException("Runtime protocol '{}' not available".format(self._protocol))

        return _runtime_protocols[self._protocol](self, self._command, log=log, **self._args)

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

    def __init__(self, tracker : Tracker):
        self._tracker = tracker

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
    def initialize(self, frame: Frame, region: Region) -> Tuple[Region, dict, float]:
        pass

    @abstractmethod
    def update(self, frame: Frame) -> Tuple[Region, dict, float]:
        pass

try:

    from vot.tracker.trax import TraxTrackerRuntime, trax_matlab_adapter, trax_python_adapter, trax_octave_adapter

    _runtime_protocols["trax"] = TraxTrackerRuntime
    _runtime_protocols["traxmatlab"] = trax_matlab_adapter
    _runtime_protocols["traxpython"] = trax_python_adapter
    _runtime_protocols["traxoctave"] = trax_octave_adapter

except OSError:
    pass

except ImportError:
    # TODO: print some kind of error
    pass

from vot.tracker.results import Trajectory, Results