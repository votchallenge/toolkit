
import os, yaml, glob

from vot import VOTException

from vot.dataset import VOTDataset, Sequence, Dataset
from vot.tracker import Tracker, Results
from vot.experiment import Experiment
from vot.stack import Stack, resolve_stack

from vot.utilities import normalize

class WorkspaceException(VOTException):
    pass

def initialize_workspace(directory, config=dict()):
    config_file = os.path.join(directory, "config.yaml")
    if os.path.isfile(config_file):
        raise WorkspaceException("Workspace already initialized")

    with open(config_file, 'w') as fp:
        yaml.dump(config, fp)

    os.makedirs(os.path.join(directory, "sequences"), exist_ok=True)
    os.makedirs(os.path.join(directory, "results"), exist_ok=True)
    os.makedirs(os.path.join(directory, "logs"), exist_ok=True)

class Workspace(object):

    def __init__(self, directory):
        config_file = os.path.join(directory, "config.yaml")
        if not os.path.isfile(config_file):
            raise WorkspaceException("Workspace not initialized")

        with open(config_file, 'r') as fp:
            self._config = yaml.load(fp, Loader=yaml.BaseLoader)

        if not "stack" in self._config:
            raise WorkspaceException("Experiment stack not found in workspace configuration")

        self._stack = resolve_stack(self._config["stack"])

        if not self._stack:
            raise WorkspaceException("Experiment stack does not exist")

        dataset_directory = normalize(self._config.get("sequences", "sequences"), directory)
        results_directory = normalize(self._config.get("results", "results"), directory)

        self._dataset = VOTDataset(dataset_directory)
        self._results = results_directory

    @property
    def registry(self):
        return self._config.get("registry", [])

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @property
    def stack(self) -> Stack:
        return self._stack

    def results(self, tracker: Tracker, experiment: Experiment, sequence: Sequence):
        root = os.path.join(self._results, os.path.join(tracker.identifier, os.path.join(experiment.identifier, sequence.name)))
        return Results(root)