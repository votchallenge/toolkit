"""Helper classes for experiments."""

from vot.dataset import Sequence
from vot.region import is_special

def _objectstart(sequence: Sequence, id: str):
    """Returns the first frame where the object appears in the sequence."""
    trajectory = sequence.object(id)
    return [x is None or is_special(x) for x in trajectory].index(False)

class MultiObjectHelper(object):
    """Helper class for multi-object sequences.

    It provides methods for querying active objects at a given frame.
    """

    def __init__(self, sequence: Sequence):
        """Initialize the helper class.

        :param sequence: The sequence to be used.
        :type sequence: Sequence
        """ 
        self._sequence = sequence
        self._ids = list(sequence.objects())
        start = [_objectstart(sequence, id) for id in self._ids]
        self._ids = sorted(zip(start, self._ids), key=lambda x: x[0])

    def new(self, position: int):
        """Returns a list of objects that appear at the given frame.

        :param position: The frame number.
        :type position: int

        :returns: A list of object ids.
        :rtype: [list]"""
        return [x[1] for x in self._ids if x[0] == position]

    def objects(self, position: int):
        """Returns a list of objects that are active at the given frame.

        :param position: The frame number.
        :type position: int

        :returns: A list of object ids.
        :rtype: [list]"""
        return [x[1] for x in self._ids if x[0] <= position]

    def all(self):
        """Returns a list of all objects in the sequence.

        :returns: A list of object ids.
        :rtype: [list]"""
        return [x[1] for x in self._ids]