
import os
import glob
import logging
from abc import ABC, abstractmethod
from datetime import datetime
import pickle

import yaml
import cachetools

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

class Storage(ABC):

    @abstractmethod
    def results(self, tracker: Tracker, experiment: Experiment, sequence: Sequence):
        pass

    @abstractmethod
    def list_results(self, registry: "Registry"):
        pass

    @abstractmethod
    def open_log(self, identifier):
        pass

    @abstractmethod
    def write(self, path, binary=False):
        pass

    @abstractmethod
    def substorage(self, name):
        pass

    @abstractmethod
    def copy(self, localfile, destination):
        pass

class LocalStorage(ABC):

    def __init__(self, root: str):
        self._root = root
        self._results = os.path.join(root, "results")

    @property
    def base(self):
        return self._root

    def results(self, tracker: Tracker, experiment: Experiment, sequence: Sequence):
        root = os.path.join(self._results, tracker.reference, experiment.identifier, sequence.name)
        return Results(root)

    def list_results(self, registry: "Registry"):
        references = [os.path.basename(x) for x in glob.glob(os.path.join(self._results, "*")) if os.path.isdir(x)]
        return registry.resolve(*references)

    def open_log(self, identifier):

        logdir = os.path.join(self.base, "logs")
        os.makedirs(logdir, exist_ok=True)

        return open(os.path.join(logdir, "{}_{:%Y-%m-%dT%H-%M-%S.%f%z}.log".format(identifier, datetime.now())), "w")

    def write(self, path, binary=False):
        if os.path.isabs(path):
            raise IOError("Only relative paths allowed")

        full = os.path.join(self.base, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)

        mode = "wb" if binary else "w"
        return open(full, mode=mode, newline="")

    def substorage(self, name):
        return LocalStorage(os.path.join(self.base, name))

    def copy(self, localfile, destination):
        import shutil
        if os.path.isabs(destination):
            raise IOError("Only relative paths allowed")

        full = os.path.join(self.base, destination)
        os.makedirs(os.path.dirname(full), exist_ok=True)

        shutil.move(localfile, os.path.join(self.base, full))
        #with open(localfile, "rb") as fin:

class Cache(cachetools.Cache):

    def __init__(self, root: str):
        super().__init__(10000)
        self._root = root
        os.makedirs(self._root, exist_ok=True)

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

    def _filename(self, key):
        if isinstance(key, tuple):
            filename = key[-1]
            if len(key) > 1:
                directory = self.directory(*key[:-1])
            else:
                directory = self.base
        else:
            filename = str(key)
            directory = self.base
        return os.path.join(directory, filename)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError as e:
            filename = self._filename(key)
            if not os.path.isfile(filename):
                raise e
            try:
                with open(filename, mode="rb") as filehandle:
                    data = pickle.load(filehandle)
                    super().__setitem__(key, data)
                    return data
            except pickle.PickleError:
                raise e

    def __setitem__(self, key, value):
        super().__setitem__(key, value)

        filename = self._filename(key)
        try:
            with open(filename, mode="wb") as filehandle:
                return pickle.dump(value, filehandle)
        except pickle.PickleError:
            pass

    def __contains__(self, key):
        filename = self._filename(key)
        return os.path.isfile(filename)

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

    def cache(self, identifier) -> Cache:
        if not isinstance(identifier, str):
            identifier = class_fullname(identifier)

        return Cache(os.path.join(self._root, "cache", identifier))