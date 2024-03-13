""" Proxy sequence classes that allow to modify the behaviour of a sequence without changing the underlying data."""
from typing import List, Set, Tuple

from vot.region import Region

from vot.dataset import Channel, Sequence, Frame

class ProxySequence(Sequence):
    """A proxy sequence base that forwards requests to undelying source sequence. Meant as a base class.
    """

    def __init__(self, source: Sequence, name: str = None):
        """Creates a proxy sequence.

        Args:
            source (Sequence): Source sequence object
        """
        if name is None:
            name = source.name
        super().__init__(name)
        self._source = source

    def __len__(self):
        """Returns the length of the sequence. Forwards the request to the source sequence.
        
        Returns:
            int: Length of the sequence.
        """
        return len(self._source)

    def frame(self, index: int) -> Frame:
        """Returns a frame object for the given index. Forwards the request to the source sequence.
        
        Args:
            index (int): Index of the frame.
            
        Returns:
            Frame: Frame object."""
        return Frame(self, index)

    def metadata(self, name, default=None):
        """Returns a metadata value for the given name. Forwards the request to the source sequence.

        Args:
            name (str): Name of the metadata.
            default (object, optional): Default value to return if the metadata is not found. Defaults to None.
        
        Returns:
            object: Metadata value.
        """
        return self._source.metadata(name, default)

    def channel(self, channel=None):
        """Returns a channel object for the given name. Forwards the request to the source sequence.

        Args:
            channel (str, optional): Name of the channel. Defaults to None.

        Returns:
            Channel: Channel object.
        """
        return self._source.channel(channel)

    def channels(self):
        """Returns a list of channel names. Forwards the request to the source sequence.

        Returns:
            list: List of channel names.
        """
        return self._source.channels()

    def objects(self):
        """Returns a list of object ids. Forwards the request to the source sequence.

        Returns:
            list: List of object ids.
        """
        return self._source.objects()

    def object(self, id, index=None):
        """Returns an object for the given id. Forwards the request to the source sequence.
        
        Args:
            id (str): Id of the object.
            index (int, optional): Index of the frame. Defaults to None.
            
        Returns:
            Object: Object object.
        """
        return self._source.object(id, index)

    def groundtruth(self, index: int = None) -> List[Region]:
        """Returns a list of groundtruth regions for the given index. Forwards the request to the source sequence.
        
        Args:
            index (int, optional): Index of the frame. Defaults to None.
            
        Returns:
            list: List of groundtruth regions.
        """
        return self._source.groundtruth(index)

    def tags(self, index=None):
        """Returns a list of tags for the given index. Forwards the request to the source sequence.
        
        Args:
            index (int, optional): Index of the frame. Defaults to None.
            
        Returns:
            list: List of tags.
        """
        return self._source.tags(index)

    def values(self, index=None):
        """Returns a list of values for the given index. Forwards the request to the source sequence.
        
        Args:
            index (int, optional): Index of the frame. Defaults to None.
        """
        return self._source.values(index)

    @property
    def size(self) -> Tuple[int, int]:
        """Returns the size of the sequence. Forwards the request to the source sequence.

        Returns:
            Tuple[int, int]: Size of the sequence.
        """
        return self._source.size


class FrameMapChannel(Channel):
    """A proxy channel that maps frames from a source channel in another order."""

    def __init__(self, source: Channel, frame_map: List[int]):
        """Creates a frame mapping proxy channel.
        
        Args:
            source (Channel): Source channel object
            frame_map (List[int]): A list of frame indices in the source channel that will form the proxy. The list is filtered
            so that all indices that are out of bounds are removed.
            
        """
        super().__init__()
        self._source = source
        self._map = frame_map

    def __len__(self):
        """Returns the length of the channel."""
        return len(self._map)

    def frame(self, index):
        """Returns a frame object for the given index.
        
        Args:
            index (int): Index of the frame.
            
        Returns:
            Frame: Frame object.
        """
        return self._source.frame(self._map[index])

    def filename(self, index):
        """Returns the filename of the frame for the given index. Index is mapped according to the frame map before the request is forwarded to the source channel.

        Args:
            index (int): Index of the frame.

        Returns:
            str: Filename of the frame.
        """
        return self._source.filename(self._map[index])

    @property
    def size(self):
        """Returns the size of the channel.
        
        Returns:
            Tuple[int, int]: Size of the channel.
        """
        return self._source.size

class FrameMapSequence(ProxySequence):
    """A proxy sequence that maps frames from a source sequence in another order.
    """

    def __init__(self, source: Sequence, frame_map: List[int]):
        """Creates a frame mapping proxy sequence.

        Args:
            source (Sequence): Source sequence object
            frame_map (List[int]): A list of frame indices in the source sequence that will form the proxy. The list is filtered
            so that all indices that are out of bounds are removed.
        """
        super().__init__(source)
        self._map = [i for i in frame_map if i >= 0 and i < len(source)]

    def channel(self, channel=None):
        """Returns a channel object for the given channel name.
        
        Args:
            channel (str): Name of the channel.
            
        Returns:
            Channel: Channel object.
        """
        sourcechannel = self._source.channel(channel)

        if sourcechannel is None:
            return None

        return FrameMapChannel(sourcechannel, self._map)

    def channels(self):
        """Returns a list of channel names.
        
        Returns:
            list: List of channel names.
        """
        return self._source.channels()

    def frame(self, index: int) -> Frame:
        """Returns a frame object for the given index. Forwards the request to the source sequence with the mapped index.
        
        Args:
            index (int): Index of the frame.
            
        Returns:
            Frame: Frame object.
        """
        return self._source.frame(self._map[index])

    def groundtruth(self, index: int = None) -> List[Region]:
        """Returns a list of groundtruth regions for the given index. Forwards the request to the source sequence with the mapped index.
        
        Args:
            index (int, optional): Index of the frame. Defaults to None.
            
        Returns:
            list: List of groundtruth regions."""
        if index is None:
            groundtruth = [None] * len(self)
            for i, m in enumerate(self._map):
                groundtruth[i] = self._source.groundtruth(m)
            return groundtruth
        else:
            return self._source.groundtruth(self._map[index])

    def object(self, id, index=None):
        """Returns an object for the given id. Forwards the request to the source sequence with the mapped index.

        Args:
            id (str): Id of the object.
            index (int, optional): Index of the frame. Defaults to None.

        Returns:
            Region: Object region or a list of object regions.
        """
        if index is None:
            groundtruth = [None] * len(self)
            for i, m in enumerate(self._map):
                groundtruth[i] = self._source.object(id, m)
            return groundtruth
        else:
            return super().object(id, self._map[index])

    def tags(self, index=None):
        """Returns a list of tags for the given index. Forwards the request to the source sequence with the mapped index.
        
        Args:
            index (int, optional): Index of the frame. Defaults to None.
            
        Returns:
            list: List of tags.
        """
        if index is None:
            # TODO: this is probably not correct
            return self._source.tags()
        else:
            return self._source.tags(self._map[index])

    def values(self, index=None):
        """Returns a list of values for the given index. Forwards the request to the source sequence with the mapped index.
        
        Args:
            index (int, optional): Index of the frame. Defaults to None.
            
        Returns:
            list: List of values.    
        """
        if index is None:
            # TODO: this is probably not correct
            return self._source.values()
        return self._source.values(self._map[index])

    def __len__(self) -> int:
        """Returns the length of the sequence. The length is the same as the length of the frame map.

        Returns:
            int: Length of the sequence.
        """
        return len(self._map)

class ChannelFilterSequence(ProxySequence):
    """A proxy sequence that only makes specific channels visible.
    """

    def __init__(self, source: Sequence, channels: Set[str]):
        """Creates a channel filter proxy sequence.
        
        Args:
            source (Sequence): Source sequence object
            channels (Set[str]): A set of channel names that will be visible in the proxy sequence. The set is filtered
            so that all channel names that are not in the source sequence are removed.
        """
        super().__init__(source)
        self._filter = [i for i in channels if i in source.channels()]

    def channel(self, channel=None):
        """Returns a channel object for the given channel name. If the channel is not in the filter, None is returned.
        
        Args:
            channel (str): Name of the channel.
            
        Returns:
            Channel: Channel object.
        """
        if channel not in self._filter:
            return None
        return self._source.channel(channel)

    def channels(self):
        """Returns a list of channel names.
        
        Returns:
            list: List of channel names.
        """
        return set(self._filter)

class ObjectFilterSequence(ProxySequence):
    """A proxy sequence that only makes specific object visible.
    """

    def __init__(self, source: Sequence, id: str, trim: bool=False):
        """Creates an object filter proxy sequence.
    
        Args:
            source (Sequence): Source sequence object
            id (str): ID of the object that will be visible in the proxy sequence.
            
        Keyword Args:
            trim (bool): If true, the sequence will be trimmed to the first and last frame where the object is visible.    
        """
        super().__init__(source, "%s_%s" % (source.name, id))
        self._id = id
        # TODO: implement trim
        self._trim = trim
    
    def objects(self):
        """Returns a dictionary of all objects in the sequence.
        
        Returns:
            Dict[str, Object]: Dictionary of all objects in the sequence.
        """
        objects = self._source.objects()
        return {self._id: objects[id]}

    def object(self, id, index=None):
        """Returns an object for the given id.
        
        Args:
            id (str): ID of the object.
            
        Returns:
            Region: Object object.  
        """
        if id != self._id:
            return None
        return self._source.object(id, index)

    def groundtruth(self, index: int = None) -> List[Region]:
        """Returns the groundtruth for the given index.
        
        Args:
            index (int): Index of the frame.
        """
        return self._source.object(self._id, index)
    
class ObjectsHideFilterSequence(ProxySequence):
    """A proxy sequence that virtually removes specified objects from the sequence. Note that the object is not removed from the sequence, but only hidden when listing them.
    """

    def __init__(self, source: Sequence, ids: Set[str]):
        """Creates an object hide filter proxy sequence.
    
        Args:
            source (Sequence): Source sequence object
            ids (Set[str]): IDs of the objects that will be hidden in the proxy sequence.
        """
        super().__init__(source)
        self._ids = ids
    
    def objects(self):
        """Returns a dictionary of all objects in the sequence.
        
        Returns:
            Dict[str, Object]: Dictionary of all objects in the sequence.
        """
        objects = self._source.objects()
        return {id for id in objects if id not in self._ids}

def IgnoreSpecialObjects(sequence: Sequence) -> Sequence:
    """Creates a proxy sequence that ignores special objects.Special objects are denoted by a 
        leading underscore in the object name. Usually, those objects are used for storing additional
        information about the sequence.
    
    Args:
        sequence (Sequence): Source sequence object.
        
    Returns:
        Sequence: Proxy sequence object.
    """

    def is_special(id: str):
        """Checks if the object id is special (starts with underscore)."""
        return id.startswith("_")
    
    ids = [id for id in sequence.objects() if is_special(id)]

    if len(ids) == 0:
        return sequence

    return ObjectsHideFilterSequence(sequence, ids)