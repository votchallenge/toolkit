
import os

from vot import VOTException

from vot.dataset import VOTDataset, Sequence, Dataset
from vot.tracker import Tracker
from vot.experiments import Experiment
from vot.stacks import Stack

class WorkspaceException(VOTException):
    pass

def initialize_workspace(directory):
    os.makedirs(os.path.join(directory, "sequences"), exist_ok=True)
    os.makedirs(os.path.join(directory, "results"), exist_ok=True)
    os.makedirs(os.path.join(directory, "logs"), exist_ok=True)

class Results(object):

    def __init__(self, root):
        self._root = root

    def exists(self, name):
        return os.path.isfile(os.path.join(self._root, name))

    def open(self, name, mode='r'):
        return open(os.path.join(self._root, name), mode)

class Workspace(object):

    def __init__(self, directory):
        dataset_directory = os.path.join(directory, "sequences")
        results_directory = os.path.join(directory, "results")
        if not os.path.isdir(dataset_directory):
            raise WorkspaceException("Workspace not initialized")
        self._dataset = VOTDataset(dataset_directory)
        self._results = results_directory

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @property
    def stack(self) -> Stack:
        return None

    def results(self, tracker: Tracker, experiment: Experiment, sequence: Sequence):
        root = os.path.join(self._results, os.path.join(tracker.identifier, os.path.join(experiment.identifier, sequence.name)))
        return Results(root)