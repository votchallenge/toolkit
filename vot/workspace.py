
import os, yaml, glob
import logging
from abc import ABC, abstractmethod
from datetime import datetime

from vot import VOTException

from vot.dataset import VOTDataset, Sequence, Dataset
from vot.tracker import Tracker, Results
from vot.experiment import Experiment
from vot.stack import Stack, resolve_stack

from vot.utilities import normalize_path, class_fullname

logger = logging.getLogger("vot")

class WorkspaceException(VOTException):
    pass

def initialize_workspace(directory, config=dict()):
    config_file = os.path.join(directory, "config.yaml")
    if os.path.isfile(config_file):
        raise WorkspaceException("Workspace already initialized")

    os.makedirs(directory, exist_ok=True)

    with open(config_file, 'w') as fp:
        yaml.dump(config, fp)

    os.makedirs(os.path.join(directory, "sequences"), exist_ok=True)
    os.makedirs(os.path.join(directory, "results"), exist_ok=True)

    if not os.path.isfile(os.path.join(directory, "trackers.ini")):
        open(os.path.join(directory, "trackers.ini"), 'w').close()

def migrate_workspace(directory):
    import re
    from numpy import genfromtxt, reshape, savetxt, all

    config_file = os.path.join(directory, "config.yaml")
    if os.path.isfile(config_file):
        raise WorkspaceException("Workspace already initialized")

    old_config_file = os.path.join(directory, "configuration.m")
    if not os.path.isfile(old_config_file):
        raise WorkspaceException("Old workspace config not detected")

    with open(old_config_file, "r") as fp:
        content = fp.read()
        matches = re.findall("set\\_global\\_variable\\('stack', '([A-Za-z0-9]+)'\\)", content)
        if not len(matches) == 1:
            raise WorkspaceException("Experiment stack could not be retrieved")
        stack = matches[0]

    for tracker_dir in [x for x in os.scandir(os.path.join(directory, "results")) if x.is_dir()]:
        for experiment_dir in [x for x in os.scandir(tracker_dir.path) if x.is_dir()]:
            for sequence_dir in [x for x in os.scandir(experiment_dir.path) if x.is_dir()]:
                timing_file = os.path.join(sequence_dir.path, "{}_time.txt".format(sequence_dir.name))
                if os.path.isfile(timing_file):
                    logger.info("Migrating %s", timing_file)
                    times = genfromtxt(timing_file, delimiter=",")
                    if len(times.shape) == 1:
                        times = reshape(times, (times.shape[0], 1))
                    for k in range(times.shape[1]):
                        if all(times[:, k] == 0):
                            break
                        savetxt(os.path.join(sequence_dir.path, \
                             "%s_%03d_time.value" % (sequence_dir.name, k+1)), \
                             times[:, k] / 1000, fmt='%.6e')
                    os.unlink(timing_file)

    try:
        resolve_stack(stack)
    except:
        logging.warning("Stack %s not found, you will have to manually edit and correct config file.", stack)

    with open(config_file, 'w') as fp:
        yaml.dump(dict(stack=stack, registry=["."]), fp)

    os.unlink(old_config_file)

    logging.info("Workspace %s migrated", directory)

class Storage(ABC):

    @abstractmethod
    def results(self, tracker: Tracker, experiment: Experiment, sequence: Sequence):
        pass

    @abstractmethod
    def list_results(self):
        pass

    @abstractmethod
    def open_log(self, identifier):
        pass

class LocalStorage(ABC):

    def __init__(self, root: str):
        self._root = root
        self._results = os.path.join(root, "results")

    @property
    def base(self):
        return self._root

    def results(self, tracker: Tracker, experiment: Experiment, sequence: Sequence):
        root = os.path.join(self._results, os.path.join(tracker.identifier, os.path.join(experiment.identifier, sequence.name)))
        return Results(root)

    def list_results(self):
        return [os.path.basename(x) for x in glob.glob(os.path.join(self._results, "*")) if os.path.isdir(x)]

    def open_log(self, identifier):

        logdir = os.path.join(self.base, "logs")
        os.makedirs(logdir, exist_ok=True)

        return open(os.path.join(logdir, "{}_{:%Y-%m-%dT%H-%M-%S.%f%z}.log".format(identifier, datetime.now())), "w")

class Cache(object):

    def __init__(self, root: str):
        self._root = root

    @property
    def base(self):
        return self._root

    def directory(self, *args):
        segments = []
        for arg in args:
            if arg is None:
                continue
            if isinstance(arg, str):
                segments.append(arg)
            elif isinstance(arg, (int, float)):
                segments.append(str(arg))
            else:
                segments.append(class_fullname(arg))

        path = os.path.join(self._root, *segments)
        os.makedirs(path, exist_ok=True)

        return path

class Workspace(object):

    def __init__(self, directory):
        directory = normalize_path(directory)
        config_file = os.path.join(directory, "config.yaml")
        if not os.path.isfile(config_file):
            raise WorkspaceException("Workspace not initialized")

        with open(config_file, 'r') as fp:
            self._config = yaml.load(fp, Loader=yaml.BaseLoader)

        if not "stack" in self._config:
            raise WorkspaceException("Experiment stack not found in workspace configuration")

        stack_file = resolve_stack(self._config["stack"], directory)

        if stack_file is None:
            raise WorkspaceException("Experiment stack does not exist")

        self._storage = LocalStorage(directory)
        self._cache = Cache(os.path.join(directory, "cache"))

        with open(stack_file, 'r') as fp:
            stack_metadata = yaml.load(fp, Loader=yaml.BaseLoader)
            self._stack = Stack(self, stack_metadata)

        dataset_directory = normalize_path(self._config.get("sequences", "sequences"), directory)

        self._download(dataset_directory)
        self._dataset = VOTDataset(dataset_directory)
        self._root = directory
        self._registry = [normalize_path(r, directory) for r in self._config.get("registry", [])]

    def _download(self, dataset_directory):
        if not os.path.exists(os.path.join(dataset_directory, "list.txt")) and not self._stack.dataset is None:
            logger.info("Stack has a dataset attached, downloading bundle '%s'", self._stack.dataset)

            from vot.dataset import download_dataset
            download_dataset(self._stack.dataset, dataset_directory)

            logger.info("Download completed")

    @property
    def directory(self):
        return self._root

    @property
    def registry(self):
        return self._registry

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @property
    def stack(self) -> Stack:
        return self._stack

    @property
    def storage(self) -> Storage:
        return self._storage

    @property
    def cache(self) -> Cache:
        return self._cache