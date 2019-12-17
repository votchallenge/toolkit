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

    def __init__(self, base):
        self._base = base

    @property
    def base(self):
        return self._base

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
        return self._sequence.channel(channel).frame(self._index)

    def filename(self, channel=None):
        return self._sequence.channel(channel).filename(self._index)

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

    def __init__(self, path):
        base, pattern = os.path.split(path)
        super().__init__(base)
        self._pattern = pattern
        self.__scan()

    def __scan(self):
        
        extension = os.path.splitext(self._pattern)[1]
        if not extension in {'.jpg', '.png'}:
            raise DatasetException("Invalid extension in pattern {}".format(self._pattern))
        
        i = 1
        self._files = []

        fullpattern = os.path.join(self.base, self._pattern)
        
        while True:
            image_file = os.path.join(fullpattern % i)

            if not os.path.isfile(image_file):
                break
            self._files.append(image_file)
            i = i + 1

        if i < 1:
            raise DatasetException("Empty sequence, no frames found.")
            
        im = cv2.imread(self._files[0])
        self._width = im.shape[1]
        self._height = im.shape[0]
        self._depth = im.shape[2]

    @property
    def length(self):
        return len(self._files)

    def frame(self, index):
        if index < 0 or index >= self.length:
            return None

        bgr = cv2.imread(self._files[index])
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def size(self):
        return self._width, self._height

    def filename(self, index):
        if index < 0 or index >= self.length:
            return None

        return self._files[index]

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

    def __init__(self, name, dataset = None):
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
    def groundtruth(self, index):
        pass

    @abstractmethod
    def tags(self, index = None):
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
        
    @property
    def path(self):
        return self._path
    
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
