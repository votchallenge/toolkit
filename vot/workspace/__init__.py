
import os
import typing
import importlib

import yaml

from attributee import Attribute, Attributee, Nested, List, String, CoerceContext

from .. import ToolkitException, get_logger
from ..dataset import Dataset, load_dataset
from ..tracker import Registry, Tracker
from ..stack import Stack, resolve_stack
from ..utilities import normalize_path, class_fullname
from ..document import ReportConfiguration
from .storage import LocalStorage, Storage, NullStorage

_logger = get_logger()

class WorkspaceException(ToolkitException):
    """Errors related to workspace raise this exception
    """
    pass

class StackLoader(Attribute):
    """Special attribute that converts a string or a dictionary input to a Stack object.
    """

    def coerce(self, value, context: typing.Optional[CoerceContext]):
        importlib.import_module("vot.analysis")
        importlib.import_module("vot.experiment")
        if isinstance(value, str):

            stack_file = resolve_stack(value, context.parent.directory)

            if stack_file is None:
                raise WorkspaceException("Experiment stack does not exist")

            with open(stack_file, 'r') as fp:
                stack_metadata = yaml.load(fp, Loader=yaml.BaseLoader)
                return Stack(value, context.parent, **stack_metadata)
        else:
            return Stack(None, context.parent, **value)

    def dump(self, value):
        if value.name is None:
            return value.dump()
        else:
            return value.name

class Workspace(Attributee):
    """Workspace class represents the main junction of trackers, datasets and experiments. Each workspace performs 
    given experiments on a provided dataset.
    """

    registry = List(String(transformer=lambda x, ctx: normalize_path(x, ctx.parent.directory)))
    stack = StackLoader()
    sequences = String(default="sequences")
    report = Nested(ReportConfiguration)

    @staticmethod
    def initialize(directory: str, config: typing.Optional[typing.Dict] = None, download: bool = True) -> None:
        """Initialize a new workspace in a given directory with the given config

        Args:
            directory (str): Root for workspace storage
            config (typing.Optional[typing.Dict], optional): Workspace initial configuration. Defaults to None.
            download (bool, optional): Download the dataset immediately. Defaults to True.

        Raises:
            WorkspaceException: When a workspace cannot be created.
        """

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
    def download_dataset(dataset: str, directory: str) -> None:
        """Download the dataset if no dataset is present already.

        Args:
            dataset (str): Dataset URL or ID
            directory (str): Directory where the dataset is saved

        """
        if os.path.exists(os.path.join(directory, "list.txt")): #TODO: this has to be improved now that we also support other datasets that may not have list.txt
            return False

        from vot.dataset import download_dataset
        download_dataset(dataset, directory)

        _logger.info("Download completed")

    @staticmethod
    def load(directory):
        """Load a workspace from a given location. This 

        Args:
            directory ([type]): [description]

        Raises:
            WorkspaceException: [description]

        Returns:
            [type]: [description]
        """
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

        Args:
            directory ([type]): [description]
        """
        self._directory = directory
        self._storage = LocalStorage(directory) if directory is not None else NullStorage()

        super().__init__(**kwargs)
        dataset_directory = normalize_path(self.sequences, directory)

        if not self.stack.dataset is None:
            Workspace.download_dataset(self.stack.dataset, dataset_directory)

        self._dataset = load_dataset(dataset_directory)

    @property
    def directory(self) -> str:
        """Returns the root directory for the workspace.

        Returns:
            str: The absolute path to the root of the workspace.
        """
        return self._directory

    @property
    def dataset(self) -> Dataset:
        """Returns dataset associated with the workspace

        Returns:
            Dataset: The dataset object.
        """
        return self._dataset

    @property
    def storage(self) -> Storage:
        """Returns the storage object associated with this workspace.

        Returns:
            Storage: The storage object.
        """
        return self._storage

    def list_results(self, registry: "Registry") -> typing.List["Tracker"]:
        """Utility method that looks for all subfolders in the results folder and tries to resolve them
        as tracker references. It returns a list of Tracker objects, i.e. trackers that have at least 
        some results or an existing results directory.

        Returns:
            [typing.List[Tracker]]: A list of trackers with results.
        """
        references = self._storage.substorage("results").folders()
        return registry.resolve(*references)
