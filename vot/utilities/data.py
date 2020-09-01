import functools


class Grid(object):

    def __init__(self, size: tuple):
        self._size = size
        self._data = [None] * functools.reduce(lambda x, y: x * y, size)

    def _ravel(self, pos):
        if not isinstance(pos, tuple):
            pos = (pos, )
        assert(len(pos) == len(self._size))
        raveled = 0
        row = 1
        for n, i in zip(reversed(self._size), reversed(pos)):
            if i < 0 or i >= n:
                raise IndexError("Index out of bounds")
            raveled = i * row + raveled
            row = row * n
        return raveled

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

    def column(self, i):
        assert self.dimensions == 2
        column = Grid((self.size()[0], ))
        for j in range(self.size()[0]):
            column[j] = self[j, i]
        return column

    def row(self, i):
        assert self.dimensions == 2
        row = Grid((self.size()[1], ))
        for j in range(self.size()[1]):
            row[j] = self[i, j]
        return row