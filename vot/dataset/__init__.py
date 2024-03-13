"""Dataset module provides an interface for accessing the datasets and sequences. It also provides a set of utility functions for downloading and extracting datasets."""

import os
import logging

from numbers import Number
from collections import namedtuple
from abc import abstractmethod, ABC
from typing import List, Mapping, Optional, Set, Tuple, Iterator

from class_registry import ClassRegistry

from PIL.Image import Image
import numpy as np

from cachetools import cached, LRUCache

from vot.region import Region
from vot import ToolkitException

import cv2

logger = logging.getLogger("vot")

dataset_downloader = ClassRegistry("vot_downloader")
sequence_indexer = ClassRegistry("vot_indexer")
sequence_reader = ClassRegistry("vot_sequence")

class DatasetException(ToolkitException):
    """Dataset and sequence related exceptions
    """
    pass

class Channel(ABC):
    """Abstract representation of individual image channel, a sequence of images with
    uniform dimensions.
    """

    def __init__(self):
        """ Base constructor for channel"""
        pass

    def __len__(self) -> int:
        """Returns the length of channel

        Returns:
            int: Length of channel
        """
        raise NotImplementedError()

    @abstractmethod
    def frame(self, index: int) -> "Frame":
        """Returns frame object for the given index

        Args:
            index (int): Index of the frame

        Returns:
            Frame: Frame object
        """
        pass

    @abstractmethod
    def filename(self, index: int) -> str:
        """Returns filename for the given index of the channel sequence
        
        Args:
            index (int): Index of the frame
            
        Returns:
            str: Filename of the frame
        """
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """Returns the size of the channel in bytes"""
        pass

class Frame(object):
    """Frame object represents a single frame in the sequence. It provides access to the
    image data, groundtruth, tags and values as a wrapper around the sequence object."""

    def __init__(self, sequence, index):
        """Base constructor for frame object
        
        Args:
            sequence (Sequence): Sequence object
            index (int): Index of the frame
            
        Returns:
            Frame: Frame object
        """
        self._sequence = sequence
        self._index = index

    @property
    def index(self) -> int:
        """Returns the index of the frame
        
        Returns:
            int: Index of the frame    
        """
        return self._index

    @property
    def sequence(self) -> 'Sequence':
        """Returns the sequence object of the frame object
        
        Returns:
            Sequence: Sequence object
        """
        return self._sequence

    def channels(self):
        """Returns the list of channels in the sequence
        
        Returns:
            List[str]: List of channels
        """
        return self._sequence.channels()

    def channel(self, channel: Optional[str] = None):
        """Returns the channel object for the given channel name
        
        Args:
            channel (Optional[str], optional): Name of the channel. Defaults to None.
        """
        channelobj = self._sequence.channel(channel)
        if channelobj is None:
            return None
        return channelobj.frame(self._index)

    def filename(self, channel: Optional[str] = None):
        """Returns the filename for the given channel name and frame index
        
        Args:
            channel (Optional[str], optional): Name of the channel. Defaults to None.
            
        Returns:
            str: Filename of the frame
        """
        channelobj = self._sequence.channel(channel)
        if channelobj is None:
            return None
        return channelobj.filename(self._index)

    def image(self, channel: Optional[str] = None) -> np.ndarray:
        """Returns the image for the given channel name and frame index
        
        Args:
            channel (Optional[str], optional): Name of the channel. Defaults to None.
        
        Returns:
            np.ndarray: Image object
        """
        channelobj = self._sequence.channel(channel)
        if channelobj is None:
            return None
        return channelobj.frame(self._index)

    def objects(self) -> List[str]:
        """Returns the list of objects in the frame

        Returns:
            List[str]: List of object ids
        """
        objects = {}
        for o in self._sequence.objects():
            region = self._sequence.object(o, self._index)
            if region is not None:
                objects[o] = region
        return objects

    def object(self, id: str) -> Region:
        """Returns the object region for the given object id and frame index
        
        Args:
            id (str): Id of the object
            
        Returns:
            Region: Object region
        """
        return self._sequence.object(id, self._index)

    def groundtruth(self) -> Region:
        """Returns the groundtruth region for the frame

        Returns:
            Region: Groundtruth region

        Raises:
            DatasetException: If groundtruth is not available
        """
        return self._sequence.groundtruth(self._index)

    def tags(self) -> List[str]:
        """Returns the tags for the frame
        
        Returns:
            List[str]: List of tags
        """
        return self._sequence.tags(self._index)

    def values(self) -> Mapping[str, float]:
        """Returns the values for the frame
        
        Returns:
            Mapping[str, float]: Mapping of values
        """
        return self._sequence.values(self._index)

class SequenceIterator(object):
    """Sequence iterator provides an iterator interface for the sequence object"""

    def __init__(self, sequence: "Sequence"):
        """Base constructor for sequence iterator
        
        Args:
            sequence (Sequence): Sequence object
        """
        self._position = 0
        self._sequence = sequence

    def __iter__(self):
        """Returns the iterator object
        
        Returns:
            SequenceIterator: Sequence iterator object
        """
        return self

    def __next__(self) -> Frame:
        """Returns the next frame object in the sequence iterator
        
        Returns:
            Frame: Frame object
        """
        if self._position >= len(self._sequence):
            raise StopIteration()
        index = self._position
        self._position += 1
        return Frame(self._sequence, index)

class InMemoryChannel(Channel):
    """In-memory channel represents a sequence of images with uniform dimensions. It is
    used to represent a sequence of images in memory."""

    def __init__(self):
        """Base constructor for in-memory channel"""
        super().__init__()
        self._images = []
        self._width = 0
        self._height = 0
        self._depth = 0

    def append(self, image):
        """Appends an image to the channel

        Args:
            image (np.ndarray): Image object
        """
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

    def __len__(self) -> int:
        """Returns the length of the sequence channel in number of frames
        
        Returns:
            int: Length of the sequence channel
        """
        return len(self._images)

    def frame(self, index):
        """Returns the frame object for the given index in the sequence channel
        
        Args:
            index (int): Index of the frame
            
        Returns:
            Frame: Frame object
        """
        if index < 0 or index >= len(self):
            return None

        return self._images[index]

    @property
    def size(self):
        """Returns the size of the channel in the format (width, height)
        
        Returns:
            Tuple[int, int]: Size of the channel
        """
        return self._width, self._height

    def filename(self, index):
        """ Thwows an exception as the sequence is available in memory and not in files. """
        raise DatasetException("Sequence is available in memory, image files not available")

class PatternFileListChannel(Channel):
    """Sequence channel implementation where each frame is stored in a file and all file names
    follow a specific pattern.
    """

    def __init__(self, path, start=1, step=1, end=None, check_files=True):
        """ Creates a new channel object
        
        Args:
            path (str): Path to the sequence
            start (int, optional): First frame index
            step (int, optional): Step between frames
            end (int, optional): Last frame index
            check_files (bool, optional): Check that files exist
            
        Raises:
            DatasetException: If the pattern is invalid
            
        Returns:
            PatternFileListChannel: Channel object
        """
        super().__init__()
        base, pattern = os.path.split(path)
        self._base = base
        self._pattern = pattern
        self.__scan(pattern, start, step, end, check_files=check_files)

    @property
    def base(self) -> str:
        """Returns the base path of the sequence

        Returns:
            str: Base path
        """
        return self._base

    @property
    def pattern(self):
        """Returns the pattern of the sequence

        Returns:
            str: Pattern
        """
        return self._pattern

    def __scan(self, pattern, start, step, end, check_files=True):
        """Scans the sequence directory for files matching the pattern and stores the file names in
        the internal list. The pattern must contain a single %d placeholder for the frame index. The
        placeholder must be at the end of the pattern. The pattern may contain a file extension. If
        the pattern contains no file extension, .jpg is assumed. If end frame is specified, the
        scanning stops when the end frame is reached. If check_files is True, and end frame is set
        then files are checked to exist.
        
        Args:
            pattern (str): Pattern
            start (int): First frame index
            step (int): Step between frames
            end (int): Last frame index
            check_files (bool, optional): Check that files exist
            
        Raises:
            DatasetException: If the pattern is invalid
        """

        extension = os.path.splitext(pattern)[1]
        if not extension in {'.jpg', '.png'}:
            raise DatasetException("Invalid extension in pattern {}".format(pattern))

        i = start
        self._files = []

        fullpattern = os.path.join(self.base, pattern)

        assert end is not None or check_files

        while True:
            image_file = os.path.join(fullpattern % i)

            if check_files and not os.path.isfile(image_file):
                break
            self._files.append(os.path.basename(image_file))
            i = i + step

            if end is not None and i > end:
                break

        if i <= start:
            raise DatasetException("Empty sequence, no frames found.")

        if os.path.isfile(self.filename(0)):
            im = cv2.imread(self.filename(0))
            self._width = im.shape[1]
            self._height = im.shape[0]
            self._depth = im.shape[2]
        else:
            self._depth = None
            self._width = None
            self._height = None

    def __len__(self) -> int:
        """Returns the number of frames in the sequence 
        
        Returns:
            int: Number of frames
        """
        return len(self._files)

    def frame(self, index: int) -> np.ndarray:
        """Returns the frame at the specified index as a numpy array. The image is loaded using
        OpenCV and converted to RGB color space if necessary.

        Args:
            index (int): Frame index

        Returns:
            np.ndarray: Frame

        Raises:
            DatasetException: If the index is out of bounds

        """
        if index < 0 or index >= len(self):
            return None

        bgr = cv2.imread(self.filename(index))

        # Check if the image is grayscale
        if len(bgr.shape) == 2:
            return bgr

        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @property
    def size(self) -> tuple:
        """Returns the size of the frames in the sequence as a tuple (width, height)

        Returns:
            tuple: Size of the frames
        """
        return self._width, self._height

    @property
    def width(self) -> int:
        """Returns the width of the frames in the sequence

        Returns:
            int: Width of the frames
        """
        return self._width

    @property
    def height(self) -> int:
        """Returns the height of the frames in the sequence
        
        Returns:
            int: Height of the frames
        """
        return self._height

    def filename(self, index) -> str:
        """Returns the filename of the frame at the specified index
        
        Args:
            index (int): Frame index
            
        Returns:
            str: Filename
        """
        if index < 0 or index >= len(self):
            return None

        return os.path.join(self.base, self._files[index])

class FrameList(object):
    """Abstract base for all sequences, just a list of frame objects
    """

    def __iter__(self):
        """Returns an iterator over the frames in the sequence
        
        Returns:
            SequenceIterator: Iterator
        """
        return SequenceIterator(self)

    def __len__(self) -> int:
        """Returns the number of frames in the sequence.
        
        Returns:
            int: Number of frames
        """
        return NotImplementedError()

    def frame(self, index: int) -> Frame:
        """Returns the frame at the specified index
        
        Args:
            index (int): Frame index
        """
        return NotImplementedError()

class Sequence(FrameList):
    """A sequence is a list of frames (multiple channels) and a list of one or more annotated objects. It also contains
    additional metadata and per-frame information, such as tags and values.
    """

    UNKNOWN = 0 # object state is unknown in this frame
    INVISIBLE = 1 # object is not visible in this frame

    def __init__(self, name: str):
        """Creates a new sequence with the specified name"""
        self._name = name

    @property
    def name(self) -> str:
        """Returns the name of the sequence
        
        Returns:
            str: Name
        """
        return self._name

    @property
    def identifier(self) -> str:
        """Returns the identifier of the sequence. The identifier is a string that uniquely identifies the sequence in
        the dataset. The identifier is usually the same as the name, but may be different if the name is not unique.
        
        Returns:
            str: Identifier
        """
        return self._name

    @abstractmethod
    def metadata(self, name, default=None):
        """Returns the value of the specified metadata field. If the field does not exist, the default value is returned
        
        Args:
            name (str): Name of the metadata field
            default (object, optional): Default value
        
        Returns:
            object: Value of the metadata field
        """
        pass

    @abstractmethod
    def channel(self, channel=None) -> Channel:
        """Returns the channel with the specified name or the default channel if no name is specified 
        
        Args:
            channel (str, optional): Name of the channel
            
        Returns:
            Channel: Channel
        """
        pass

    @abstractmethod
    def channels(self) -> Set[str]:
        """Returns the names of all channels in the sequence
        
        Returns:
            set: Names of all channels
        """
        pass

    @abstractmethod
    def objects(self) -> Set[str]:
        """Returns the names of all objects in the sequence
        
        Returns:
            set: Names of all objects
        """
        pass

    @abstractmethod
    def object(self, id, index=None):
        """Returns the object with the specified name or identifier. If the index is specified, the object is returned
        only if it is visible in the frame at the specified index.
        
        Args:
            id (str): Name or identifier of the object
            index (int, optional): Frame index
            
        Returns:
            Region: Object
        """
        pass

    @abstractmethod
    def groundtruth(self, index: int) -> Region:
        """Returns the ground truth region for the specified frame index or None if no ground truth is available for the
        frame or the frame index is out of bounds. This is a legacy method for compatibility with single-object datasets
        and should not be used in new code.

        Args:
            index (int): Frame index

        Returns:
            Region: Ground truth region
        """
        pass

    @abstractmethod
    def tags(self, index=None) -> List[str]:
        """Returns the tags for the specified frame index or None if no tags are available for the frame or the frame 
        index is out of bounds.
        
        Args:
            index (int, optional): Frame index
            
        Returns:
            list: List of tags
        """
        pass

    @abstractmethod
    def values(self, index=None) -> Mapping[str, Number]:
        """Returns the values for the specified frame index or None if no values are available for the frame or the frame
        index is out of bounds.
        
        Args:
            index (int, optional): Frame index
            
        Returns:
            dict: Dictionary of values"""
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        """Returns the width of the frames in the sequence in pixels

        Returns:
            int: Width of the frames
        """
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        """Returns the height of the frames in the sequence in pixels

        Returns:
            int: Height of the frames
        """
        pass

    @property
    def size(self) -> Tuple[int, int]:
        """Returns the size of the frames in the sequence in pixels as a tuple (width, height)
        
        Returns:
            tuple: Size of the frames
        """
        return self.width, self.height

    def describe(self):
        """Returns a dictionary with information about the sequence
        
        Returns:
            dict: Dictionary with information
        """
        return dict(length=len(self), width=self.width, height=self.height)

class Dataset(object):
    """Base abstract class for a tracking dataset, a list of image sequences addressable by their names and interatable.
    """

    def __init__(self, sequences: Mapping[str, Sequence]) -> None:
        """Creates a new dataset with the specified sequences
        
        Args:
            sequences (dict): Dictionary of sequences
        """
        self._sequences = sequences

    def __len__(self) -> int:
        """Returns the number of sequences in the dataset
        
        Returns:
            int: Number of sequences
        """
        return len(self._sequences)

    def __getitem__(self, key: str) -> Sequence:
        """Returns the sequence with the specified name
        
        Args:
            key (str): Sequence name
            
        Returns:
            Sequence: Sequence
        """
        return self._sequences[key]

    def __contains__(self, key: str) -> bool:
        """Returns true if the dataset contains a sequence with the specified name
        
        Args:
            key (str): Sequence name
            
        Returns:
            bool: True if the dataset contains the sequence
        """
        return key in self._sequences

    def __iter__(self) -> Iterator[Sequence]:
        """Returns an iterator over the sequences in the dataset
        
        Returns:
            DatasetIterator: Iterator
        """
        return iter(self._sequences.values())

    def list(self) -> List[str]:
        """Returns a list of unique sequence names

        Returns:
            List[str]: List of sequence names
        """
        return list(self._sequences.keys())

    def keys(self) -> List[str]:
        """Returns a list of unique sequence names

        Returns:
            List[str]: List of sequence names
        """
        return list(self._sequences.keys())

SequenceData = namedtuple("SequenceData", ["channels", "objects", "tags", "values", "length"])

from vot import config

@cached(LRUCache(maxsize=config.sequence_cache_size))
def _cached_loader(sequence):
    """Loads the sequence data from the sequence object. This function serves as a cache for the sequence data and is only
    called if the sequence data is not already loaded. The cache is implemented as a LRU cache with a maximum size
    specified in the configuration."""
    return sequence._loader(sequence._metadata)

class BasedSequence(Sequence):
    """This class implements the caching of the sequence data. The sequence data is loaded only when it is needed.
    """

    def __init__(self, name: str, loader: callable, metadata: dict = None):
        """Initializes the sequence
        
        Args:
            name (str): Sequence name
            loader (callable): Loader function that takes the metadata as an argument and returns a SequenceData object
            metadata (dict, optional): Sequence metadata. Defaults to None.
            
        Raises:
            ValueError: If the loader is not callable    
        """
        super().__init__(name)
        self._loader = loader
        self._metadata = metadata if metadata is not None else {}

    def __preload(self):
        """Loads the sequence data if needed. This is an internal function that should not be called directly.
        It calles a cached loader function that is implemented as a LRU cache with a configurable maximum size."""
        return _cached_loader(self)

    def metadata(self, name: str, default: object=None) -> object:
        """Returns the metadata value with the specified name.
        
        Args:
            name (str): Metadata name
            default (object, optional): Default value. Defaults to None.
            
        Returns:
            object: Metadata value"""
        return self._metadata.get(name, default)

    def channels(self) -> List[str]:
        """Returns a list of channel names in the sequence
        
        Returns:
            List[str]: List of channel names
        """
        data = self.__preload()
        return data.channels.keys()

    def channel(self, channel: str=None) -> Channel:
        """Returns the channel with the specified name. If the channel name is not specified, the default channel is returned.

        Args:
            channel (str, optional): Channel name. Defaults to None.

        Returns:
            Channel: Channel
        """
        data = self.__preload()
        if channel is None:
            channel = self.metadata("channel.default")
        return data.channels.get(channel, None)

    def frame(self, index):
        """Returns the frame with the specified index in the sequence as a Frame object
        
        Args:
            index (int): Frame index
            
        Returns:
            Frame: Frame
        """
        return Frame(self, index)

    def objects(self) -> List[str]:
        """Returns a list of object ids in the sequence 
        
        Returns:
            List[str]: List of object ids
        """
        data = self.__preload()
        return data.objects.keys()

    def object(self, id, index=None) -> Region:
        """Returns the object with the specified id. If the index is specified, the object is returned as a Region object.

        Args:
            id (str): Object id
            index (int, optional): Frame index. Defaults to None.

        Returns:
            Region: Object region
        """
        data = self.__preload()
        obj = data.objects.get(id, None)
        if index is None:
            return obj
        if obj is None:
            return None
        return obj[index]

    def groundtruth(self, index=None):
        """Returns the groundtruth object. If the index is specified, the object is returned as a Region object. If the
        sequence contains more than one object, an exception is raised. If more objects are present, this method 
        ignores special objects.
        
        Args:
            index (int, optional): Frame index. Defaults to None.
            
        Returns:
            Region: Groundtruth region
        """
        objids = self.objects()
   
        if len(objids) != 1:
            # Filter special objects first
            objids = [o for o in objids if not o.startswith("_")]
            if len(objids) != 1:
                raise DatasetException("More than one object in sequence")

        id = next(iter(objids))
        return self.object(id, index)

    def tags(self, index: int = None) -> List[str]:
        """Returns a list of tags in the sequence. If the index is specified, only the tags that are present in the frame 
        with the specified index are returned.
        
        Args:
            index (int, optional): Frame index. Defaults to None.
            
        Returns:
            List[str]: List of tags
        """
        data = self.__preload()
        if index is None:
            return data.tags.keys()
        return [t for t, sq in data.tags.items() if sq[index]]

    def values(self, index: int = None) -> List[float]:
        """Returns a list of values in the sequence. If the index is specified, only the values that are present in the frame
        with the specified index are returned.
        
        Args:
            index (int, optional): Frame index. Defaults to None.
            
        Returns:
            List[float]: List of values
        """
        data = self.__preload()
        if index is None:
            return data.values.keys()
        return {v: sq[index] for v, sq in data.values.items()}

    @property
    def size(self):
        """Returns the sequence size as a tuple (width, height)
        
        Returns:
            tuple: Sequence size
            
        """
        return self.width, self.height

    @property
    def width(self):
        """Returns the sequence width"""
        return self._metadata["width"]

    @property
    def height(self):
        """Returns the sequence height"""
        return self._metadata["height"]

    def __len__(self):
        """Returns the sequence length in frames
        
        Returns:
            int: Sequence length
        """
        data = self.__preload()
        return data.length

class InMemorySequence(Sequence):
    """ An in-memory sequence that can be used to construct a sequence programmatically and store it do disk.
    Used mainly for testing and debugging.
    
    Only single object sequences are supported at the moment.
    """

    def __init__(self, name, channels):
        """Creates a new in-memory sequence.
        
        Args:
            name (str): Sequence name
            channels (list): List of channel names
            
        Raises:
            DatasetException: If images are not provided for all channels
        """
        super().__init__(name)
        self._channels = {c: InMemoryChannel() for c in channels}
        self._tags = {}
        self._values = {}
        self._groundtruth = []

    def append(self, images: dict, region: "Region", tags: list = None, values: dict = None):
        """Appends a new frame to the sequence. The frame is specified by a dictionary of images, a region and optional
        tags and values.
        
        Args:
            images (dict): Dictionary of images
            region (Region): Region
            tags (list, optional): List of tags
            values (dict, optional): Dictionary of values
        """

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
                self._tags[tag] = [False] * len(self)
            self._tags[tag].append(True)
        for tag in set(self._tags.keys()).difference(tags):
                self._tags[tag].append(False)

        if values is None:
            values = dict()
        for name, value in values.items():
            if not name in self._values:
                self._values[name] = [0] * len(self)
            self._values[tag].append(value)
        for name in set(self._values.keys()).difference(values.keys()):
                self._values[name].append(0)

        self._groundtruth.append(region)

    def channel(self, channel : str) -> "Channel":
        """Returns the specified channel object.
        
        Args:
            channel (str): Channel name
            
        Returns:
            Channel: Channel object
        """
        return self._channels.get(channel, None)
    
    def channels(self) -> List[str]:
        """Returns a list of channel names.

        Returns:
            List[str]: List of channel names

        """
        return set(self._channels.keys())
    
    def frame(self, index : int) -> "Frame":
        """Returns the specified frame. The frame is returned as a Frame object.
        
        Args:
            index (int): Frame index
            
        Returns:
            Frame: Frame object
        """
        return Frame(self, index)
    
    def groundtruth(self, index: int = None) -> "Region":
        """Returns the groundtruth object. If the index is specified, the object is returned as a Region object. If the
        sequence contains more than one object, an exception is raised. If the index is not specified, the groundtruth
        object is returned as a Region object. If the sequence contains more than one object, an exception is raised.
        
        Args:
            index (int, optional): Frame index. Defaults to None.
            
        Returns:
            Region: Groundtruth object
        """
        if index is None:
            return self._groundtruth
        return self._groundtruth[index]

    def object(self, id: str, index: int = None) -> "Region":
        """Returns the specified object. If the index is specified, the object is returned as a Region object. If the
        sequence contains more than one object, an exception is raised. If the index is not specified, the groundtruth
        object is returned as a Region object. If the sequence contains more than one object, an exception is raised.
        
        Args:
            id (str): Object id
            index (int, optional): Frame index. Defaults to None.
            
        Returns:
            Region: Object
        """
        if id != "object":
            return None

        if index is None:
            return self._groundtruth
        return self._groundtruth[index]

    def objects(self, index: str = None) -> List[str]:
        """Returns a list of object ids. If the index is specified, only the objects that are present in the frame with
        the specified index are returned.

        Since only single object sequences are supported, the only object id that is returned is "object".
        
        Args:
            index (int, optional): Frame index. Defaults to None."""
        return ["object"]

    def tags(self, index=None):
        """Returns a list of tags in the sequence. If the index is specified, only the tags that are present in the frame
        with the specified index are returned.
        
        Args:
            index (int, optional): Frame index. Defaults to None.
            
        Returns:
            List[str]: List of tags
        """
        if index is None:
            return self._tags.keys()
        return [t for t, sq in self._tags.items() if sq[index]]
    
    def values(self, index=None):
        """Returns a list of values in the sequence. If the index is specified, only the values that are present in the
        frame with the specified index are returned.
        
        Args:
            index (int, optional): Frame index. Defaults to None.
            
        Returns:
            List[str]: List of values
        """
        if index is None:
            return self._values.keys()
        return {v: sq[index] for v, sq in self._values.items()}
    
    def __len__(self):
        """Returns the sequence length in frames
        
        Returns:
            int: Sequence length
        """
        return len(self._groundtruth)
    
    @property
    def width(self) -> int:
        """Returns the sequence width
        
        Returns:
            int: Sequence width
        """
        return self.channel().width
    
    @property
    def height(self) -> int:
        """Returns the sequence height
        
        Returns:
            int: Sequence height
        """
        return self.channel().height
    
    @property
    def size(self) -> tuple:
        """Returns the sequence size as a tuple (width, height) 
        
        Returns:
            tuple: Sequence size
        """
        return self.channel().size

    def channels(self) -> list:
        """Returns a list of channel names 
        
        Returns:
            list: List of channel names
        """
        return set(self._channels.keys())

def download_bundle(url: str, path: str = "."):
    """Downloads a dataset bundle as a ZIP file and decompresses it.

    Args:
        url (str): Source bundle URL
        path (str, optional): Destination directory. Defaults to ".".

    Raises:
        DatasetException: If the bundle cannot be downloaded or is not supported.
    """

    from vot.utilities.net import download_uncompress, NetworkException

    if not url.endswith(".zip"):
        raise DatasetException("Unknown bundle format")

    logger.info('Downloading sequence bundle from "%s". This may take a while ...', url)

    try:
        download_uncompress(url, path)
    except NetworkException as e:
        raise DatasetException("Unable do download dataset bundle, Please try to download the bundle manually from {} and uncompress it to {}'".format(url, path))
    except IOError as e:
        raise DatasetException("Unable to extract dataset bundle, is the target directory writable and do you have enough space?")

def download_dataset(url: str, path: str):
    """Downloads a dataset from a given url or an alias.

    Args:
        url (str): URL to the data bundle or metadata description file
        path (str): Destination directory

    Raises:
        DatasetException: If the dataset is not found or a network error occured
    """
    from urllib.parse import urlsplit

    try:
        res = urlsplit(url)

        if res.scheme in ["http", "https"]:
            if res.path.endswith(".json"):
                from .common import download_dataset_meta
                download_dataset_meta(url, path)
                return
            else:
                download_bundle(url, path)
                return

        raise DatasetException("Unknown dataset domain: {}".format(res.scheme))

    except ValueError:

        if url in dataset_downloader:
            dataset_downloader[url](path)
            return

        raise DatasetException("Illegal dataset identifier: {}".format(url))


def load_dataset(path: str) -> Dataset:
    """Loads a dataset from a local directory

    Args:
        path (str): The path to the local dataset data

    Raises:
        DatasetException: When a folder does not exist or the format is not recognized.

    Returns:
        Dataset: Dataset object
    """

    from collections import OrderedDict

    sequence_list = None

    for _, indexer in sequence_indexer.items():
        logger.debug("Attempting to index sequences with {}.{}".format(indexer.__module__, indexer.__name__))
        sequence_list = indexer(path)
        if sequence_list is not None:
            break
        
    if sequence_list is None or len(sequence_list) == 0:
        raise DatasetException("Unable to locate sequences in {}".format(path))

    sequences = OrderedDict()

    logger.debug("Loading sequences...")

    for sequence_id in sequence_list:
        sequence_path = sequence_id.strip()
        if not os.path.isabs(sequence_id):
            sequence_path = os.path.join(path, sequence_id)
        sequence = load_sequence(sequence_path)
        sequences[sequence.name] = sequence

    logger.debug("Found %d sequences in dataset" % len(sequence_list))

    return Dataset(sequences)

def load_sequence(path: str) -> Sequence:
    """Loads a sequence from a given path (directory), tries to guess the format of the sequence.

    Args:
        path (str): The path to the local sequence data

    Raises:
        DatasetException: If an loading error occures, unsupported format or other issues.

    Returns:
        Sequence: Sequence object
    """

    for _, loader in sequence_reader.items():
        logger.debug("Attempting to load sequence with {}.{}".format(loader.__module__, loader.__name__))
        sequence = loader(path)
        if sequence is not None:
            return sequence

    raise DatasetException("Unable to load sequence, unknown format or unsupported sequence: {}".format(path))

import importlib
for module in [".common", ".otb", ".got10k", ".trackingnet"]:
    importlib.import_module(module, package="vot.dataset")

# Legacy reader is registered last, otherwise it will cause problems
# TODO: implement explicit ordering of readers
@sequence_reader.register("legacy")
def read_legacy_sequence(path: str) -> Sequence:
    """Wrapper around the legacy sequence reader."""
    from vot.dataset.common import read_sequence_legacy
    return read_sequence_legacy(path)