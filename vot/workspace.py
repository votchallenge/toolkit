
import os
import glob
import logging
from abc import ABC, abstractmethod
from datetime import datetime
import pickle
import importlib

import yaml
import cachetools

from vot import VOTException

from vot.dataset import VOTDataset, Sequence, Dataset
from vot.tracker import Tracker, Results
from vot.experiment import Experiment
from vot.stack import Stack, resolve_stack

from vot.utilities import normalize_path, class_fullname
from vot.utilities.attributes import Attribute, Attributee, Nested, List, String
from vot.document import ReportConfiguration

logger = logging.getLogger("vot")

class WorkspaceException(VOTException):
    pass

class Storage(ABC):

    @abstractmethod
    def results(self, tracker: Tracker, experiment: Experiment, sequence: Sequence):
        pass

    @abstractmethod
    def documents(self):
        pass

    @abstractmethod
    def folders(self):
        pass

    @abstractmethod
    def write(self, name, binary=False):
        pass

    @abstractmethod
    def read(self, name, binary=False):
        pass

    @abstractmethod
    def isdocument(self, name):
        pass

    @abstractmethod
    def isfolder(self, name):
        pass

    @abstractmethod
    def substorage(self, name):
        pass

    @abstractmethod
    def copy(self, localfile, destination):
        pass

class VoidStorage(Storage):

    def results(self, tracker: Tracker, experiment: Experiment, sequence: Sequence):
        return Results(self)

    def write(self, name, binary=False):
        if binary:
            return open(os.devnull, "wb")
        else:
            return open(os.devnull, "w")

    def documents(self):
        return []

    def folders(self):
        return []

    def read(self, name, binary=False):
        return None

    def isdocument(self, name):
        return False

    def isfolder(self, name):
        return False

    def substorage(self, name):
        return VoidStorage()

    def copy(self, localfile, destination):
        return

class LocalStorage(Storage):

    def __init__(self, root: str):
        self._root = root
        self._results = os.path.join(root, "results")

    @property
    def base(self):
        return self._root

    def results(self, tracker: Tracker, experiment: Experiment, sequence: Sequence):
        storage = LocalStorage(os.path.join(self._results, tracker.reference, experiment.identifier, sequence.name))
        return Results(storage)

    def documents(self):
        return [name for name in os.listdir(self._root) if os.path.isfile(os.path.join(self._root, name))]

    def folders(self):
        return [name for name in os.listdir(self._root) if os.path.isdir(os.path.join(self._root, name))]

    def write(self, name, binary=False):
        if os.path.isabs(name):
            raise IOError("Only relative paths allowed")

        full = os.path.join(self.base, name)
        os.makedirs(os.path.dirname(full), exist_ok=True)

        if binary:
            return open(full, mode="wb")
        else:
            return open(full, mode="w", newline="")

    
    def read(self, name, binary=False):
        if os.path.isabs(name):
            raise IOError("Only relative paths allowed")

        full = os.path.join(self.base, name)

        if binary:
            return open(full, mode="rb")
        else:
            return open(full, mode="r", newline="")

    def isdocument(self, name):
        return os.path.isfile(os.path.join(self._root, name))

    def isfolder(self, name):
        return os.path.isdir(os.path.join(self._root, name))

    def substorage(self, name):
        return LocalStorage(os.path.join(self.base, name))

    def copy(self, localfile, destination):
        import shutil
        if os.path.isabs(destination):
            raise IOError("Only relative paths allowed")

        full = os.path.join(self.base, destination)
        os.makedirs(os.path.dirname(full), exist_ok=True)

        shutil.move(localfile, os.path.join(self.base, full))

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

class Cache(cachetools.Cache):

    def __init__(self, storage: LocalStorage):
        super().__init__(10000)
        self._storage = storage

    def _filename(self, key):
        if isinstance(key, tuple):
            filename = key[-1]
            if len(key) > 1:
                directory = self._storage.directory(*key[:-1])
            else:
                directory = self._storage.base
        else:
            filename = str(key)
            directory = self._storage.base
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

class StackLoader(Attribute):

    def coerce(self, value, ctx):
        importlib.import_module("vot.analysis")
        importlib.import_module("vot.experiment")
        if isinstance(value, str):

            stack_file = resolve_stack(value, ctx["parent"].directory)

            if stack_file is None:
                raise WorkspaceException("Experiment stack does not exist")

            with open(stack_file, 'r') as fp:
                stack_metadata = yaml.load(fp, Loader=yaml.BaseLoader)
                return Stack(value, ctx["parent"], **stack_metadata)
        else:
            return Stack(None, ctx["parent"], **value)

    def dump(self, value):
        if value.name is None:
            return value.dump()
        else:
            return value.name

class Workspace(Attributee):

    registry = List(String(transformer=lambda x, ctx: normalize_path(x, ctx["parent"].directory)))
    stack = StackLoader()
    sequences = String(default="sequences")
    report = Nested(ReportConfiguration)

    @staticmethod
    def initialize(directory, config=None, download=True):
        config_file = os.path.join(directory, "config.yaml")
        if os.path.isfile(config_file):
            raise WorkspaceException("Workspace already initialized")

        os.makedirs(directory, exist_ok=True)

        with open(config_file, 'w') as fp:
            yaml.dump(config if config is not None else dict(), fp)

        os.makedirs(os.path.join(directory, "sequences"), exist_ok=True)
        os.makedirs(os.path.join(directory, "results"), exist_ok=True)

        if not os.path.isfile(os.path.join(directory, "trackers.ini")):
            open(os.path.join(directory, "trackers.ini"), 'w').close()

        if download:
            # Try do retrieve dataset from stack and download it
            stack_file = resolve_stack(config["stack"], directory)
            dataset_directory = normalize_path(config.get("sequences", "sequences"), directory)
            if stack_file is None:
                return
            dataset = None
            with open(stack_file, 'r') as fp:
                stack_metadata = yaml.load(fp, Loader=yaml.BaseLoader)
                dataset = stack_metadata["dataset"]
            if dataset:
                Workspace.download_dataset(dataset, dataset_directory)

    @staticmethod
    def download_dataset(dataset, directory):
        if os.path.exists(os.path.join(directory, "list.txt")):
            return False

        from vot.dataset import download_dataset
        download_dataset(dataset, directory)

        logger.info("Download completed")

    @staticmethod
    def load(directory):
        directory = normalize_path(directory)
        config_file = os.path.join(directory, "config.yaml")
        if not os.path.isfile(config_file):
            raise WorkspaceException("Workspace not initialized")

        with open(config_file, 'r') as fp:
            config = yaml.load(fp, Loader=yaml.BaseLoader)

            return Workspace(directory, **config)

    def __init__(self, directory, **kwargs):
        self._directory = directory
        self._storage = LocalStorage(directory) if directory is not None else VoidStorage()
        super().__init__(**kwargs)
        dataset_directory = normalize_path(self.sequences, directory)

        if not self.stack.dataset is None:
            Workspace.download_dataset(self.stack.dataset, dataset_directory)

        self._dataset = VOTDataset(dataset_directory)

    @property
    def directory(self) -> str:
        return self._directory

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @property
    def storage(self) -> LocalStorage:
        return self._storage

    def cache(self, identifier) -> LocalStorage:
        if not isinstance(identifier, str):
            identifier = class_fullname(identifier)

        return self._storage.substorage("cache").substorage(identifier)

    def list_results(self, registry: "Registry"):
        references = self._storage.substorage("results").folders()
        return registry.resolve(*references)
