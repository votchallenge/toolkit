"""This module contains the Workspace class that represents the main junction of
trackers, datasets and experiments."""

import os
import typing
import importlib

import yaml
from lazy_object_proxy import Proxy

from attributee import Attribute, Attributee, Nested, List, String, CoerceContext

from .. import ToolkitException, get_logger
from ..dataset import Dataset, load_dataset
from ..tracker import Registry, Tracker
from ..stack import Stack, resolve_stack
from ..utilities import normalize_path
from ..report import ReportConfiguration
from .storage import LocalStorage, Storage, NullStorage

_logger = get_logger()

class WorkspaceException(ToolkitException):
    """Errors related to workspace raise this exception."""
    pass

class StackLoader(Attribute):
    """Special attribute that converts a string or a dictionary input to a Stack
    object."""

    def coerce(self, value, context: typing.Optional[CoerceContext]):
        """Coerce a value to a Stack object.

        :param value: Value to coerce
        :type value: typing.Any
        :param context: Coercion context
        :type context: typing.Optional[CoerceContext]

        :returns: Coerced value
        :rtype: Stack"""
        importlib.import_module("vot.analysis")
        importlib.import_module("vot.experiment")
        if isinstance(value, str):

            stack_file = resolve_stack(value, context.parent.directory)

            if stack_file is None:
                raise WorkspaceException("Experiment stack does not exist")

            stack = Stack.read(stack_file)
            stack._name = value

            return stack
        else:
            return Stack(**value)

    def dump(self, value: "Stack") -> str:
        """Dump a Stack object to a string or a dictionary.

        :param value: Value to dump
        :type value: Stack

        :returns: Dumped value
        :rtype: str"""
        if value.name is None:
            return value.dump()
        else:
            return value.name

class RegistryLoader(Attribute):
    """Special attribute that converts a list of strings input to a Registry object.

    The paths are appended to the global registry search paths.
    """
    
    def coerce(self, value, context: typing.Optional[CoerceContext]):
        
        from vot import config, get_logger
        
        # Workspace registry paths are relative to the workspace directory
        paths = list(List(String(transformer=lambda x, ctx: normalize_path(x, ctx.parent.directory))).coerce(value, context))

        # Combine the paths with the global registry search paths (relative to the current directory)
        registry = Registry(paths + [normalize_path(x, os.curdir) for x in config.registry], root=context.parent.directory)
        registry._paths = paths
 
        get_logger().debug("Found data for %d trackers", len(registry))

        return registry

    def dump(self, value: "Registry") -> typing.List[str]:
        assert isinstance(value, Registry)
        return value._paths

class Workspace(Attributee):
    """Workspace class represents the main junction of trackers, datasets and
    experiments.

    Each workspace performs given experiments on a provided dataset.
    """

    registry = RegistryLoader() # List(String(transformer=lambda x, ctx: normalize_path(x, ctx.parent.directory)))
    stack = StackLoader()
    sequences = String(default="sequences")
    report = Nested(ReportConfiguration)

    @staticmethod
    def exists(directory: str) -> bool:
        """Check if a workspace exists in a given directory.

        :param directory: Directory to check
        :type directory: str

        :returns: True if the workspace exists, False otherwise.
        :rtype: bool"""
        return os.path.isfile(os.path.join(directory, "config.yaml"))

    @staticmethod
    def initialize(directory: str, config: typing.Optional[typing.Dict] = None, download: bool = True) -> None:
        """Initialize a new workspace in a given directory with the given config.

        :param directory: Root for workspace storage
        :type directory: str
        :param config: Workspace initial configuration. Defaults to None.
        :type config: typing.Optional[typing.Dict], optional
        :param download: Download the dataset immediately. Defaults to True.
        :type download: bool, optional

        :raises WorkspaceException: When a workspace cannot be created."""

        config_file = os.path.join(directory, "config.yaml")
        if Workspace.exists(directory):
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
    def download_dataset(dataset: str, directory: str) -> None:
        """Download the dataset if no dataset is present already.

        :param dataset: Dataset URL or ID
        :type dataset: str
        :param directory: Directory where the dataset is saved
        :type directory: str
        """
        if os.path.exists(os.path.join(directory, "list.txt")): #TODO: this has to be improved now that we also support other datasets that may not have list.txt
            return False

        from vot.dataset import download_dataset
        download_dataset(dataset, directory)

        _logger.info("Download completed")

    @staticmethod
    def load(directory):
        """Load a workspace from a given location. This.

        :param directory: [description]
        :type directory: [type]

        :raises WorkspaceException: [description]
        :returns: [description]
        :rtype: [type]"""
        directory = normalize_path(directory)
        config_file = os.path.join(directory, "config.yaml")
        if not os.path.isfile(config_file):
            raise WorkspaceException("Workspace not initialized")

        with open(config_file, 'r') as fp:
            config = yaml.load(fp, Loader=yaml.BaseLoader)
            return Workspace(directory, **config)

    def __init__(self, directory: str, **kwargs):
        """Do not call this constructor directly unless you know what you are doing,
        instead use the static Workspace.load method.

        :param directory: [description]
        :type directory: [type]
        """
        self._directory = directory

        self._storage = Proxy(lambda: LocalStorage(directory) if directory is not None else NullStorage())
        
        super().__init__(**kwargs)

        dataset_directory = normalize_path(self.sequences, directory)

        if not self.stack.dataset is None:
            Workspace.download_dataset(self.stack.dataset, dataset_directory)

        self._dataset = load_dataset(dataset_directory)

        # Register storage with all experiments in the stack
        for experiment in self.stack.experiments.values():
            experiment._storage = self._storage

    @property
    def directory(self) -> str:
        """Returns the root directory for the workspace.

        :returns: The absolute path to the root of the workspace.
        :rtype: str"""
        return self._directory

    @property
    def dataset(self) -> Dataset:
        """Returns dataset associated with the workspace.

        :returns: The dataset object.
        :rtype: Dataset"""
        return self._dataset

    @property
    def storage(self) -> Storage:
        """Returns the storage object associated with this workspace.

        :returns: The storage object.
        :rtype: Storage"""
        return self._storage

    def list_results(self, registry: "Registry") -> typing.List["Tracker"]:
        """Utility method that looks for all subfolders in the results folder and tries
        to resolve them as tracker references. It returns a list of Tracker objects,
        i.e. trackers that have at least some results or an existing results directory.

        :returns: A list of trackers with results.
        :rtype: [typing.List[Tracker]]"""
        references = self._storage.substorage("results").folders()
        return registry.resolve(*references)
