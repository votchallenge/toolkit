
import os
import glob
import logging
from collections import OrderedDict
import configparser

import six

from vot import get_logger
from vot.dataset import Dataset, DatasetException, BaseSequence, PatternFileListChannel
from vot.region import Special
from vot.region.io import read_trajectory
from vot.utilities import Progress

logger = get_logger()

def load_channel(source):

    extension = os.path.splitext(source)[1]

    if extension == '':
        source = os.path.join(source, '%08d.jpg')
    return PatternFileListChannel(source)

class GOT10kSequence(BaseSequence):

    def __init__(self, base, name=None, dataset=None):
        self._base = base
        if name is None:
            name = os.path.basename(base)
        super().__init__(name, dataset)

    @staticmethod
    def check(path: str):
        return os.path.isfile(os.path.join(path, 'groundtruth.txt')) and not os.path.isfile(os.path.join(path, 'sequence'))

    def _read_metadata(self):
        metadata = dict(fps=30, format="default")

        if os.path.isfile(os.path.join(self._base, 'meta_info.ini')):
            config = configparser.ConfigParser()
            config.read(os.path.join(self._base, 'meta_info.ini'))
            metadata.update(config["METAINFO"])
            metadata["fps"] = int(metadata["anno_fps"][:-3])

        metadata["channel.default"] = "color"

        return metadata

    def _read(self):

        channels = {}
        tags = {}
        values = {}
        groundtruth = []

        channels["color"] = load_channel(os.path.join(self._base, "%08d.jpg"))
        self._metadata["channel.default"] = "color"
        self._metadata["width"], self._metadata["height"] = six.next(six.itervalues(channels)).size

        groundtruth_file = os.path.join(self._base, self.metadata("groundtruth", "groundtruth.txt"))
        groundtruth = read_trajectory(groundtruth_file)

        if len(groundtruth) == 1 and channels["color"].length > 1:
            # We are dealing with testing dataset, only first frame is available, so we pad the
            # groundtruth with unknowns. Only unsupervised experiment will work, but it is ok
            groundtruth.extend([Special(Special.UNKNOWN)] * (channels["color"].length - 1))

        self._metadata["length"] = len(groundtruth)

        tagfiles = glob.glob(os.path.join(self._base, '*.label'))

        for tagfile in tagfiles:
            with open(tagfile, 'r') as filehandle:
                tagname = os.path.splitext(os.path.basename(tagfile))[0]
                tag = [line.strip() == "1" for line in filehandle.readlines()]
                while not len(tag) >= len(groundtruth):
                    tag.append(False)
                tags[tagname] = tag

        valuefiles = glob.glob(os.path.join(self._base, '*.value'))

        for valuefile in valuefiles:
            with open(valuefile, 'r') as filehandle:
                valuename = os.path.splitext(os.path.basename(valuefile))[0]
                value = [float(line.strip()) for line in filehandle.readlines()]
                while not len(value) >= len(groundtruth):
                    value.append(0.0)
                values[valuename] = value

        for name, channel in channels.items():
            if not channel.length == len(groundtruth):
                raise DatasetException("Length mismatch for channel %s" % name)

        for name, tag in tags.items():
            if not len(tag) == len(groundtruth):
                tag_tmp = len(groundtruth) * [False]
                tag_tmp[:len(tag)] = tag
                tag = tag_tmp

        for name, value in values.items():
            if not len(value) == len(groundtruth):
                raise DatasetException("Length mismatch for value %s" % name)

        return channels, groundtruth, tags, values

class GOT10kDataset(Dataset):

    def __init__(self, path, sequence_list="list.txt"):
        super().__init__(path)

        if not os.path.isabs(sequence_list):
            sequence_list = os.path.join(path, sequence_list)

        if not os.path.isfile(sequence_list):
            raise DatasetException("Sequence list does not exist")

        with open(sequence_list, 'r') as handle:
            names = handle.readlines()

        self._sequences = OrderedDict()

        with Progress("Loading dataset", len(names)) as progress:

            for name in names:
                self._sequences[name.strip()] = GOT10kSequence(os.path.join(path, name.strip()), dataset=self)
                progress.relative(1)

    @staticmethod
    def check(path: str):
        if not os.path.isfile(os.path.join(path, 'list.txt')):
            return False

        with open(os.path.join(path, 'list.txt'), 'r') as handle:
            sequence = handle.readline().strip()
            return GOT10kSequence.check(os.path.join(path, sequence))

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
