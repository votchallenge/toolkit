
import os
import glob
import logging
from collections import OrderedDict

import six

from vot.dataset import Dataset, DatasetException, BaseSequence, PatternFileListChannel
from vot.region import Special
from vot.region.io import read_trajectory
from vot.utilities import Progress

logger = logging.getLogger("vot")

def load_channel(source):

    extension = os.path.splitext(source)[1]

    if extension == '':
        source = os.path.join(source, '%d.jpg')
    return PatternFileListChannel(source)

class TrackingNetSequence(BaseSequence):

    def __init__(self, base, dataset=None):
        self._base = base
        name = os.path.splitext(os.path.basename(base))[0]
        super().__init__(name, dataset)

    @staticmethod
    def check(path: str):
        root = os.path.dirname(os.path.dirname(path))
        name = os.path.splitext(os.path.basename(path))[0]
        return os.path.isfile(path) and os.path.isdir(os.path.join(root, 'frames', name))

    def _read_metadata(self):
        metadata = dict(fps=30)
        metadata["channel.default"] = "color"
        return {}

    def _read(self):

        channels = {}
        tags = {}
        values = {}
        groundtruth = []

        root = os.path.dirname(os.path.dirname(self._base))

        channels["color"] = load_channel(os.path.join(root, 'frames', self.name))
        self._metadata["channel.default"] = "color"
        self._metadata["width"], self._metadata["height"] = six.next(six.itervalues(channels)).size

        groundtruth = read_trajectory(self._base)

        if len(groundtruth) == 1 and channels["color"].length > 1:
            # We are dealing with testing dataset, only first frame is available, so we pad the
            # groundtruth with unknowns. Only unsupervised experiment will work, but it is ok
            groundtruth.extend([Special(Special.UNKNOWN)] * (channels["color"].length - 1))

        self._metadata["length"] = len(groundtruth)

        return channels, groundtruth, tags, values

class TrackingNetDataset(Dataset):

    def __init__(self, path, splits=False):
        super().__init__(path)

        if not splits and not TrackingNetDataset.check(path):
            raise DatasetException("Unsupported dataset format, expected TrackingNet")

        sequences = []
        if not splits:
            for file in glob.glob(os.path.join(path, "anno", "*.txt")):
                sequences.append(file)
        else:
            # Special mode to load all training splits 
            for split in ["TRAIN_%d" % i for i in range(0, 12)]:
                for file in glob.glob(os.path.join(path, split, "anno", "*.txt")):
                    sequences.append(file)

        self._sequences = OrderedDict()

        with Progress("Loading dataset", len(sequences)) as progress:
            for sequence in sequences:
                name = os.path.splitext(os.path.basename(sequence))[0]
                self._sequences[name] = TrackingNetSequence(sequence, dataset=self)
                progress.relative(1)

    @staticmethod
    def check(path: str):

        for dirname in ["anno", "frames"]:
            if not os.path.isdir(os.path.join(path, dirname)):
                return False

        return True

    @property
    def path(self):
        return self._path

    @property
    def length(self):
        return len(self._sequences)

    def __getitem__(self, key):
        return self._sequences[key]

    def __contains__(self, key):
        return key in self._sequences

    def __iter__(self):
        return self._sequences.values().__iter__()

    def list(self):
        return list(self._sequences.keys())
