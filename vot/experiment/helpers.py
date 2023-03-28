

from vot.dataset import Sequence, Frame
from vot.region import RegionType

def _objectstart(sequence: Sequence, id: str):
    trajectory = sequence.object(id)
    return [x is None or x.type == RegionType.SPECIAL for x in trajectory].index(False)

class MultiObjectHelper(object):

    def __init__(self, sequence: Sequence):
        self._sequence = sequence
        self._ids = list(sequence.objects())
        start = [_objectstart(sequence, id) for id in self._ids]
        self._ids = sorted(zip(start, self._ids), key=lambda x: x[0])

    def new(self, position: int):
        return [x[1] for x in self._ids if x[0] == position]

    def objects(self, position: int):
        return [x[1] for x in self._ids if x[0] <= position]

    def all(self):
        return [x[1] for x in self._ids]