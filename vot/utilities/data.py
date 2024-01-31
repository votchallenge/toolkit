""" Data structures for storing data in a grid."""

import functools
import unittest

class Grid(object):
    """ A grid is a multidimensional array with named dimensions. """

    @staticmethod
    def scalar(obj):
        """ Creates a grid with a single cell containing the given object.
         
        Args:
            obj (object): The object to store in the grid.
        """
        grid = Grid(1,1)
        grid[0, 0] = obj
        return grid

    def __init__(self, *size):
        """ Creates a grid with the given dimensions.
        
        Args:
            size (int): The size of each dimension.
        """
        assert len(size) > 0
        self._size = size
        self._data = [None] * functools.reduce(lambda x, y: x * y, size)

    def _ravel(self, pos):
        """ Converts a multidimensional index to a single index. 
        
        Args:
            pos (tuple): The multidimensional index.
            
        Returns:
            int: The single index.
        """
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
        """ Converts a single index to a multidimensional index.

        Args:
            index (int): The single index.

        Returns:
            tuple: The multidimensional index.
        """
        unraveled = []
        for n in reversed(self._size):
            unraveled.append(index % n)
            index = index // n
        return tuple(reversed(unraveled))

    def __str__(self):
        """ Returns a string representation of the grid."""
        return str(self._data)

    @property
    def dimensions(self):
        """ Returns the number of dimensions of the grid. """
        return len(self._size)

    def size(self, i: int = None):
        """ Returns the size of the grid or the size of a specific dimension. 
        
        Args:
            i (int): The dimension to query. If None, the size of the grid is returned.
            
        Returns:
            int: The size of the grid or the size of the given dimension.
        """
        if i is None:
            return tuple(self._size)
        assert i >= 0 and i < len(self._size)
        return self._size[i]

    def __len__(self):
        """ Returns the number of elements in the grid. """
        return len(self._data)

    def __getitem__(self, i):
        """ Returns the element at the given index.
        
        Args:
            i (tuple): The index of the element. If the grid is one-dimensional, the index can be an integer.
        
        Returns:
            object: The element at the given index.
        """
        return self._data[self._ravel(i)]

    def __setitem__(self, i, data):
        """ Sets the element at the given index. 

        Args:
            i (tuple): The index of the element. If the grid is one-dimensional, the index can be an integer.
            data (object): The data to store at the given index.
        """
        self._data[self._ravel(i)] = data

    def __iter__(self):
        """ Returns an iterator over the elements of the grid. """
        return iter(self._data)

    def cell(self, *i):
        """ Returns the element at the given index packed in a scalar grid.
        
        Args:
            i (int): The index of the element. If the grid is one-dimensional, the index can be an integer.
            
        Returns:
            object: The element at the given index packed in a scalar grid.
        """
        return Grid.scalar(self[i])

    def column(self, i):
        """ Returns the column at the given index. 
        
        Args:
            i (int): The index of the column.
            
        Returns:
            Grid: The column at the given index.
        """
        assert self.dimensions == 2
        column = Grid(1, self.size()[0])
        for j in range(self.size()[0]):
            column[0, j] = self[j, i]
        return column

    def row(self, i):
        """ Returns the row at the given index.
        
        Args:
            i (int): The index of the row.
            
        Returns:
            Grid: The row at the given index.
        """
        assert self.dimensions == 2
        row = Grid(self.size()[1], 1)
        for j in range(self.size()[1]):
            row[j, 0] = self[i, j]
        return row

    def foreach(self, cb) -> "Grid":
        """ Applies a function to each element of the grid.
        
        Args:
            cb (function): The function to apply to each element. The first argument is the element, the following
                arguments are the indices of the element.
                
        Returns:
            Grid: A grid containing the results of the function.
        """
        result = Grid(*self._size)

        for i, x in enumerate(self._data):
            a = self._unravel(i)
            result[a] = cb(x, *a)

        return result

class TestGrid(unittest.TestCase):
    """ Unit tests for the Grid class. """

    def test_foreach1(self):
        """ Tests the foreach method. """

        a = Grid(5, 3)

        b = a.foreach(lambda x, i, j: 5)

        self.assertTrue(all([x == 5 for x in b]), "Output incorrect")

    def test_foreach2(self):
        """ Tests the foreach method. """

        a = Grid(5, 6, 3)

        b = a.foreach(lambda x, i, j, k: k)

        reference = [x % 3 for x in range(len(a))]

        self.assertListEqual(list(b), reference)
