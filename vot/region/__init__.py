"""
Region description classes.
"""

import os
import sys

from copy import copy
from functools import reduce
from abc import abstractmethod, ABC
from typing import Tuple
from enum import Enum

import numpy as np

from vot import VOTException

class ConversionException(VOTException):
    """Region conversion exception, the conversion cannot be performed
    """
    pass

class RegionType(Enum):
    """Enumeration of region types
    """
    SPECIAL = 0
    RECTANGLE = 1
    POLYGON = 2
    MASK = 3

def parse(string):
    tokens = [float(t) for t in string.split(',')]
    if len(tokens) == 1:
        return Special(tokens[0])
    if len(tokens) == 4:
        return Rectangle(tokens[0], tokens[1], tokens[2], tokens[3])
    elif len(tokens) % 2 == 0 and len(tokens) > 4:
        return Polygon([(tokens[i], tokens[i+1]) for i in range(0, len(tokens), 2)])
    return None

class Region(ABC):
    """
    Base class for all region containers

    :var type: type of the region
    """
    def __init__(self):
        pass

    @abstractmethod
    def type(self):
        pass

    @abstractmethod
    def copy(self):
        """Copy region to another object
        """

    @abstractmethod
    def convert(self, rtype: RegionType):
        """Convert region to another type. Note that some conversions
        degrade information.
        Arguments:
            rtype {RegionType} -- Desired type.
        """


class Special(Region):
    """
    Special region

    :var code: Code value
    """
    def __init__(self, code):
        """ Constructor

        :param code: Special code
        """
        super().__init__()
        self._code = int(code)

    def __str__(self):
        """ Create string from class """
        return '{}'.format(self._code)

    def type(self):
        return RegionType.SPECIAL

    def copy(self):
        return Special(self._code)

    def convert(self, rtype: RegionType):
        if rtype == RegionType.SPECIAL:
            return self.copy()
        else:
            raise ConversionException("Unable to convert special region to {}".format(rtype))

    def code(self):
        """Retiurns special code for this region
        Returns:
            int -- Type code
        """
        return self._code

class Rectangle(Region):
    """
    Rectangle region

    :var x: top left x coord of the rectangle region
    :var float y: top left y coord of the rectangle region
    :var float w: width of the rectangle region
    :var float h: height of the rectangle region
    """
    def __init__(self, x=0, y=0, width=0, height=0):
        """ Constructor

            :param float x: top left x coord of the rectangle region
            :param float y: top left y coord of the rectangle region
            :param float w: width of the rectangle region
            :param float h: height of the rectangle region
        """
        super().__init__()
        self.x, self.y, self.width, self.height = x, y, width, height

    def __str__(self):
        """ Create string from class """
        return '{},{},{},{}'.format(self.x, self.y, self.width, self.height)

    def type(self):
        return RegionType.RECTANGLE

    def copy(self):
        return copy(self)

    def convert(self, rtype: RegionType):
        if rtype == RegionType.RECTANGLE:
            return self.copy()
        elif rtype == RegionType.POLYGON:
            points = []
            points.append((self.x, self.y))
            points.append((self.x + self.width, self.y))
            points.append((self.x + self.width, self.y + self.height))
            points.append((self.x, self.y + self.height))
            return Polygon(points)

        elif rtype == RegionType.MASK:
            return Mask(np.ones((self.height, self.width), np.uint8), (self.x, self.y))
        else:
            raise ConversionException("Unable to convert rectangle region to {}".format(rtype))

    def draw(self, handle, color, width=1):
        handle.line([(self.x, self.y), (self.x + self.width, self.y)], color, width)
        handle.line([(self.x + self.width, self.y), (self.x + self.width, self.y + self.height)],
                    color, width)
        handle.line([(self.x + self.width, self.y + self.height), (self.x, self.y + self.height)],
                    color, width)
        handle.line([(self.x, self.y + self.height), (self.x, self.y)], color, width)

    def resize(self, factor=1):
        return Rectangle(self.x * factor, self.y * factor,
                         self.width * factor, self.height * factor)

    def center(self):
        return (self.x + self.width / 2, self.y + self.height / 2)

class Polygon(Region):
    """
    Polygon region

    :var list points: List of points as tuples [(x1,y1), (x2,y2),...,(xN,yN)]
    :var int count: number of points
    """
    def __init__(self, points):
        """
        Constructor

        :param list points: List of points as tuples [(x1,y1), (x2,y2),...,(xN,yN)]
        """
        super().__init__()
        assert isinstance(points, list)
        # do not allow empty list
        assert points
        assert reduce(lambda x, y: x and y, [isinstance(p, tuple) for p in points])
        self.count = len(points)
        self.points = points

    def __str__(self):
        """ Create string from class """
        return ','.join(['{},{}'.format(p[0], p[1]) for p in self.points])

    def type(self):
        return RegionType.POLYGON

    def copy(self):
        return copy(self)

    def convert(self, rtype: RegionType):
        if rtype == RegionType.POLYGON:
            return self.copy()
        elif rtype == RegionType.RECTANGLE:
            top = sys.float_info.max
            bottom = -sys.float_info.max
            left = sys.float_info.max
            right = -sys.float_info.max

            for point in self.points:
                top = min(top, point[1])
                bottom = max(bottom, point[1])
                left = min(left, point[0])
                right = max(right, point[0])

            return Rectangle(left, top, right - left, bottom - top)
        #elif region.type() == RegionType.MASK:
        #    return
        else:
            raise ConversionException("Unable to convert polygon region to {}".format(rtype))


    def draw(self, handle, color, width=1):
        handle.line(self.points, color, width)
        handle.line([self.points[0], self.points[-1]], color, width)

    def resize(self, factor=1):
        return Polygon([(p[0] * factor, p[1] * factor) for p in self.points])

def mask2bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (cmin, rmin, cmax, rmax)

class Mask(Region):
    """Mask region


    """

    def __init__(self, mask: np.array, offset: Tuple[int, int] = (0, 0)):
        super().__init__()
        self.mask = mask.astype(np.uint8)
        self.mask[self.mask != 0] = 255
        self.offset = offset
        self._optimize()

    def _optimize(self):
        bounds = mask2bbox(self.mask)
        self.mask = self.mask[bounds[1]:bounds[3], bounds[0]:bounds[2]]
        self.offset = (bounds[0], bounds[1])

    def type(self):
        return RegionType.MASK

    def copy(self):
        return copy(self)

    def convert(self, rtype: RegionType):
        if rtype == RegionType.MASK:
            return self.copy()
        elif rtype == RegionType.RECTANGLE:
            bounds = mask2bbox(self.mask)

            return Rectangle(bounds[0] + self.offset[0], bounds[1] + self.offset[1],
                            bounds[2] - bounds[0], bounds[3] - bounds[1])
        else:
            raise ConversionException("Unable to convert mask region to {}".format(rtype))

