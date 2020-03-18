
from typing import List

from vot.dataset import Channel, Sequence, Frame

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

class FrameMapSequence(Sequence):

    def __init__(self, source: Sequence, frame_map: List[int]):
        super().__init__(source.name, source.dataset)
        self._source = source
        self._map = [i for i in frame_map if i >= 0 and i < source.length]

    def __len__(self):
        return self.length

    def frame(self, index):
        return Frame(self, index)

    def metadata(self, name, default=None):
        return self._source.metadata(name, default)

    def channel(self, channel=None):
        sourcechannel = self._source.channel(channel)

        if sourcechannel is None:
            return None

        return FrameMapChannel(sourcechannel, self._map)

    def channels(self):
        return self._source.channels()

    def groundtruth(self, index=None):
        if index is None:
            groundtruth = [None] * len(self)
            for i, m in enumerate(self._map):
                groundtruth[i] = self._source.groundtruth(m)
            return groundtruth
        else:
            return self._source.groundtruth(self._map[index])

    def tags(self, index=None):
        if index is None:
            return self._source.tags()
        else:
            return self._source.tags(self._map[index])

    def values(self, index=None):
        if index is None:
            return self._source.values()
        return self._source.values(self._map[index])

    @property
    def size(self):
        return self._source.size

    @property
    def length(self):
        return len(self._map)
