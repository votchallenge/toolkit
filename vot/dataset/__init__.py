import os
import json
import glob

from abc import abstractmethod, ABC

from PIL.Image import Image
import numpy as np

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

    def image(self, channel=None):
        channelobj = self._sequence.channel(channel)
        if channelobj is None:
            return None
        return channelobj.frame(self._index)

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

class InMemoryChannel(Channel):

    def __init__(self):
        super().__init__()
        self._images = []
        self._width = 0
        self._height = 0
        self._depth = 0

    def append(self, image):
        if isinstance(image, Image):
            image = np.asarray(image)

        if len(image.shape) == 3:
            height, width, depth = image.shape
        elif len(image.shape) == 2:
            height, width = image.shape
            depth = 1
        else:
            raise DatasetException("Illegal image dimensions")

        if self._width > 0:
            if not (self._width == width and self._height == height):
                raise DatasetException("Size of images does not match")
            if not self._depth == depth:
                raise DatasetException("Channels of images does not match")
        else:
            self._width = width
            self._height = height
            self._depth = depth

        self._images.append(image)

    @property
    def length(self):
        return len(self._images)

    def frame(self, index):
        if index < 0 or index >= self.length:
            return None

        return self._images[index]

    @property
    def size(self):
        return self._width, self._height

    def filename(self, index):
        raise DatasetException("Sequence is available in memory, image files not available")

class PatternFileListChannel(Channel):

    def __init__(self, path, start=1, step=1):
        super().__init__()
        base, pattern = os.path.split(path)
        self._base = base
        self._pattern = pattern
        self.__scan(pattern, start, step)

    @property
    def base(self):
        return self._base

    @property
    def pattern(self):
        return self._pattern

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

class BaseSequence(Sequence):

    def __init__(self, name, dataset=None):
        super().__init__(name, dataset)
        self._metadata = {}
        self._channels = {}
        self._tags = {}
        self._values = {}
        self._groundtruth = []
    
    def metadata(self, name, default=None):
        return self._metadata.get(name, default)

    def channels(self):
        return self._channels

    def channel(self, channel=None):
        if channel is None:
            channel = self.metadata("channel.default")
        return self._channels.get(channel, None)

    def frame(self, index):
        return Frame(self, index)

    def groundtruth(self, index=None):
        if index is None:
            return self._groundtruth
        return self._groundtruth[index]

    def tags(self, index=None):
        if index is None:
            return self._tags.keys()
        return [t for t, sq in self._tags.items() if sq[index]]

    def values(self, index=None):
        if index is None:
            return self._values.keys()
        return {v: sq[index] for v, sq in self._values.items()}

    @property
    def size(self):
        return self.channel().size

    @property
    def length(self):
        return len(self._groundtruth)

class InMemorySequence(BaseSequence):

    def __init__(self, name, channels):
        super().__init__(name, None)
        self._channels = {c: InMemoryChannel() for c in channels}

    def append(self, images: dict, region: "Region", tags: list = None, values: dict = None):

        if not set(images.keys()).issuperset(self._channels.keys()):
            raise DatasetException("Images not provided for all channels")

        for k, channel in self._channels.items():
            channel.append(images[k])

        if tags is None:
            tags = set()
        else:
            tags = set(tags)
        for tag in tags:
            if not tag in self._tags:
                self._tags[tag] = [False] * self.length
            self._tags[tag].append(True)
        for tag in set(self._tags.keys()).difference(tags):
                self._tags[tag].append(False)

        if values is None:
            values = dict()
        for name, value in values.items():
            if not name in self._values:
                self._values[name] = [0] * self.length
            self._values[tag].append(value)
        for name in set(self._values.keys()).difference(values.keys()):
                self._values[name].append(0)

        self._groundtruth.append(region)


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
