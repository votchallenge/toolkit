import os
import json
import glob

from abc import abstractmethod, ABC

from vot import VOTException
from vot.utilities import read_properties
from vot.region import parse

import cv2

class DatasetException(VOTException):
    pass

class Channel(ABC):

    def __init__(self):
        pass

    @property
    @abstractmethod
    def length(self):
        pass

    @abstractmethod
    def frame(self, index):
        pass

    @abstractmethod
    def filename(self, index):
        pass

    @property
    @abstractmethod
    def size(self):
        pass

class Frame(object):

    def __init__(self, sequence, index):
        self._sequence = sequence
        self._index = index

    @property
    def index(self) -> int:
        return self._index

    @property
    def sequence(self) -> 'Sequence':
        return self._sequence

    def channels(self):
        return self._sequence.channels()

    def channel(self, channel=None):
        channelobj = self._sequence.channel(channel)
        if channelobj is None:
            return None
        return channelobj.frame(self._index)

    def filename(self, channel=None):
        channelobj = self._sequence.channel(channel)
        if channelobj is None:
            return None
        return channelobj.filename(self._index)

    def groundtruth(self):
        return self._sequence.groundtruth(self._index)

    def tags(self, index = None):
        return self._sequence.tags(self._index)

    def values(self, index=None):
        return self._sequence.values(self._index)

class SequenceIterator(object):

    def __init__(self, sequence):
        self._position = 0
        self._sequence = sequence

    def __iter__(self):
        return self

    def __next__(self):
        if self._position >= len(self._sequence):
            raise StopIteration()
        index = self._position
        self._position += 1
        return Frame(self._sequence, index)

class PatternFileListChannel(Channel):

    def __init__(self, path, start=1, step=1):
        super().__init__()
        base, pattern = os.path.split(path)
        self._base = base
        self.__scan(pattern, start, step)

    @property
    def base(self):
        return self._base

    def __scan(self, pattern, start, step):

        extension = os.path.splitext(pattern)[1]
        if not extension in {'.jpg', '.png'}:
            raise DatasetException("Invalid extension in pattern {}".format(pattern))

        i = start
        self._files = []

        fullpattern = os.path.join(self.base, pattern)

        while True:
            image_file = os.path.join(fullpattern % i)

            if not os.path.isfile(image_file):
                break
            self._files.append(os.path.basename(image_file))
            i = i + step

        if i <= start:
            raise DatasetException("Empty sequence, no frames found.")

        im = cv2.imread(self.filename(0))
        self._width = im.shape[1]
        self._height = im.shape[0]
        self._depth = im.shape[2]

    @property
    def length(self):
        return len(self._files)

    def frame(self, index):
        if index < 0 or index >= self.length:
            return None

        bgr = cv2.imread(self.filename(index))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @property
    def size(self):
        return self._width, self._height

    def filename(self, index):
        if index < 0 or index >= self.length:
            return None

        return os.path.join(self.base, self._files[index])

class FrameList(ABC):

    def __iter__(self):
        return SequenceIterator(self)

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def frame(self, index):
        pass

class Sequence(FrameList):

    def __init__(self, name: str, dataset: "Dataset" = None):
        self._name = name
        self._dataset = dataset

    def __len__(self):
        return self.length

    @property
    def name(self):
        return self._name

    @property
    def dataset(self):
        return self._dataset

    @abstractmethod
    def metadata(self, name, default=None):
        pass

    @abstractmethod
    def channel(self, channel=None):
        pass

    @abstractmethod
    def channels(self):
        pass

    @abstractmethod
    def groundtruth(self, index: int):
        pass

    @abstractmethod
    def tags(self, index=None):
        pass

    @abstractmethod
    def values(self, index=None):
        pass

    @property
    @abstractmethod
    def size(self):
        pass

    @property
    @abstractmethod
    def length(self):
        pass

class Dataset(ABC):

    def __init__(self, path):
        self._path = path

    def __len__(self):
        return self.length

    @property
    def path(self):
        return self._path

    @property
    @abstractmethod
    def length(self):
        pass

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __hasitem__(self, key):
        return False

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def list(self):
        return []

from .vot import VOTDataset, VOTSequence

from .vot import download_dataset as download_vot_dataset

def download_dataset(identifier: str, path: str):

    split = identifier.find(":")
    domain = "vot"

    if split > 0:
        domain = identifier[0:split].lower()
        identifier = identifier[split+1:]

    if domain == "vot":
        download_vot_dataset(identifier, path)
    else:
        raise DatasetException("Unknown dataset domain: {}".format(domain))
