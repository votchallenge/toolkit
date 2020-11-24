import functools
from typing import Tuple, Union, Iterable
import unittest

import numpy as np

class Grid(object):

    @staticmethod
    def scalar(obj):
        grid = Grid(1,1)
        grid[0, 0] = obj
        return grid

    def __init__(self, *size):
        assert len(size) > 0
        self._size = size
        self._data = [None] * functools.reduce(lambda x, y: x * y, size)

    def _ravel(self, pos):
        if not isinstance(pos, tuple):
            pos = (pos, )
        assert len(pos) == len(self._size)

        raveled = 0
        row = 1
        for n, i in zip(reversed(self._size), reversed(pos)):
            if i < 0 or i >= n:
                raise IndexError("Index out of bounds")
            raveled = i * row + raveled
            row = row * n
        return raveled

    def _unravel(self, index):
        unraveled = []
        for n in reversed(self._size):
            unraveled.append(index % n)
            index = index // n
        return tuple(reversed(unraveled))

    def __str__(self):
        return str(self._data)

    @property
    def dimensions(self):
        return len(self._size)

    def size(self, i: int = None):
        if i is None:
            return tuple(self._size)
        assert i >= 0 and i < len(self._size)
        return self._size[i]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[self._ravel(i)]

    def __setitem__(self, i, data):
        self._data[self._ravel(i)] = data

    def __iter__(self):
        return iter(self._data)

    def cell(self, *i):
        return Grid.scalar(self[i])

    def column(self, i):
        assert self.dimensions == 2
        column = Grid(1, self.size()[0])
        for j in range(self.size()[0]):
            column[0, j] = self[j, i]
        return column

    def row(self, i):
        assert self.dimensions == 2
        row = Grid(self.size()[1], 1)
        for j in range(self.size()[1]):
            row[j, 0] = self[i, j]
        return row

    def foreach(self, cb) -> "Grid":
        result = Grid(*self._size)

        for i, x in enumerate(self._data):
            a = self._unravel(i)
            result[a] = cb(x, *a)

        return result

class TestGrid(unittest.TestCase):

    def test_foreach1(self):

        a = Grid(5, 3)

        b = a.foreach(lambda x, i, j: 5)

        self.assertTrue(all([x == 5 for x in b]), "Output incorrect")

    def test_foreach2(self):

        a = Grid(5, 6, 3)

        b = a.foreach(lambda x, i, j, k: k)

        reference = [x % 3 for x in range(len(a))]

        self.assertListEqual(list(b), reference)
