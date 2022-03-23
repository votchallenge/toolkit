
from typing import List, Set, Tuple

from trax import Region

from vot.dataset import Channel, Sequence, Frame

class ProxySequence(Sequence):
    """A proxy sequence base that forwards requests to undelying source sequence. Meant as a base class.
    """

    def __init__(self, source: Sequence):
        """Creates a proxy sequence.

        Args:
            source (Sequence): Source sequence object
        """
        super().__init__(source.name, source.dataset)
        self._source = source

    
    def __len__(self):
        return self.length

    def frame(self, index: int) -> Frame:
        return Frame(self, index)

    def metadata(self, name, default=None):
        return self._source.metadata(name, default)

    def channel(self, channel=None):
        return self._source.channel(channel)

    def channels(self):
        return self._source.channels()

    def groundtruth(self, index: int = None) -> List[Region]:
        return self._source.groundtruth(index)

    def tags(self, index=None):
        return self._source.tags(index)

    def values(self, index=None):
        return self._source.values(index)

    @property
    def size(self) -> Tuple[int, int]:
        return self._source.size

    @property
    def length(self) -> int:
        return len(self._source)


class FrameMapChannel(Channel):

    def __init__(self, source: Channel, frame_map: List[int]):
        super().__init__()
        self._source = source
        self._map = frame_map

    @property
    def length(self):
        return len(self._map)

    def frame(self, index):
        return self._source.frame(self._map[index])

    def filename(self, index):
        return self._source.filename(self._map[index])

    @property
    def size(self):
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
        self._map = [i for i in frame_map if i >= 0 and i < source.length]

    
    def __len__(self):
        return self.length

    def channel(self, channel=None):
        sourcechannel = self._source.channel(channel)

        if sourcechannel is None:
            return None

        return FrameMapChannel(sourcechannel, self._map)

    def channels(self):
        return self._source.channels()

    def groundtruth(self, index: int = None) -> List[Region]:
        if index is None:
            groundtruth = [None] * len(self)
            for i, m in enumerate(self._map):
                groundtruth[i] = self._source.groundtruth(m)
            return groundtruth
        else:
            return self._source.groundtruth(self._map[index])

    def tags(self, index=None):
        if index is None:
            # TODO: this is probably not correct
            return self._source.tags()
        else:
            return self._source.tags(self._map[index])

    def values(self, index=None):
        if index is None:
            # TODO: this is probably not correct
            return self._source.values()
        return self._source.values(self._map[index])

    @property
    def length(self) -> int:
        return len(self._map)

class ChannelFilterSequence(ProxySequence):
    """A proxy sequence that only makes specific channels visible.
    """

    def __init__(self, source: Sequence, channels: Set[str]):
        super().__init__(source)
        self._filter = [i for i in channels if i in source.channels()]

    def channel(self, channel=None):
        if channel not in self._filter:
            return None

        sourcechannel = self._source.channel(channel)

        if sourcechannel is None:
            return None

        return FrameMapChannel(sourcechannel, self._map)

    def channels(self):
        return list(self._channels)
