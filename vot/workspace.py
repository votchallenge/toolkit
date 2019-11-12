
import os

from vot import VOTException

from vot.dataset import VOTDataset
from vot.tracker import Tracker

class WorkspaceException(VOTException):
    pass

def initialize_workspace(directory):
    os.makedirs(os.path.join(directory, "sequences"), exist_ok=True)
    os.makedirs(os.path.join(directory, "results"), exist_ok=True)
    os.makedirs(os.path.join(directory, "logs"), exist_ok=True)


class Workspace(object):

    def __init__(self, directory):
        dataset_directory = os.path.join(directory, "sequences")
        if not os.path.isdir(dataset_directory):
            raise WorkspaceException("Workspace not initialized")
        self._dataset = VOTDataset(dataset_directory)

    @property
    def dataset(self):
        return self._dataset

    def results(self, sequence, tracker: Tracker, result=None, repeat=0):
        return self._dataset