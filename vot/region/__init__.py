"""This module contains classes for region representation and manipulation.

Regions are also used to represent results of trackers as well as groundtruth
trajectories. The module also contains functions for calculating overlaps between
regions and for converting between different region types.
"""

from abc import abstractmethod, ABC
from enum import Enum

from vot import ToolkitException
from vot.utilities.draw import DrawHandle

class RegionException(ToolkitException):
    """General region exception."""

class ConversionException(RegionException):
    """Region conversion exception, the conversion cannot be performed."""
    def __init__(self, *args, source=None):
        """Constructor.

        :param *args: Arguments for the base exception

        :param source: Source region (default: {None})
        :type source: Region
        """
        super().__init__(*args)
        self._source = source

class Region(ABC):
    """Base class for all region containers."""
    def __init__(self):
        """Base constructor."""
        super().__init__()

    @abstractmethod
    def copy(self):
        """Copy region to another object.

        :returns: Region -- Copy of the region"""
        raise NotImplementedError

class Special(Region):
    """Special region, meaning of the code can change depending on the context.

    :var code: Code value
    """

    def __init__(self, code):
        """Constructor.

        :param code: Code value
        :type code: int
        """
        super().__init__()
        self._code = int(code)

    def __str__(self):
        """Create string from class."""
        return '{}'.format(self._code)

    @staticmethod
    def convert(region: Region):
        """Convert region to special region. Note that some conversions degrade
        information.

        :param region: Region to convert
        :type region: Region

        :raises ConversionException: Unable to convert region to special region
        :returns: Special -- Converted region"""
        if isinstance(region, Special):
            return region.copy()
        else:
            raise ConversionException("Unable to convert {} region to special region".format(region.type), source=region)

    @property
    def code(self):
        """Retiurns special code for this region.

        :returns: int -- Type code"""
        return self._code

    def draw(self, handle: DrawHandle):
        """Draw region to the image using the provided handle.

        :param handle: Draw handle
        :type handle: DrawHandle
        """
        pass

    def is_empty(self):
        """Check if region is empty.

        Special regions are always empty by definition.
        """
        return True

    def copy(self):
        """Create a copy of the special region."""
        return Special(self._code)

class Point(Region):
    """Special region, meaning of the code can change depending on the context.

    :var code: Code value
    """

    def __init__(self, x, y):
        """Constructor.

        :param x: X coordinate
        :type x: float
        :param y: Y coordinate
        :type y: float
        """
        super().__init__()
        self._x = float(x)
        self._y = float(y)

    def __str__(self):
        """Create string from class."""
        return '{},{}'.format(self._x, self._y)

    def copy(self):
        """Copy region to another object."""
        return Point(self._x, self._y)

    @staticmethod
    def convert(region: Region):
        """Convert region to point region. Note that some conversions degrade
        information.

        :param region: Region to convert
        :type region: Region

        :raises ConversionException: Unable to convert region to point region
        :returns: Point -- Converted region"""
        if isinstance(region, Point):
            return region.copy()
        else:
            raise ConversionException("Unable to convert {} region to point region".format(region.type), source=region)

    @property
    def x(self):
        """Retiurns X coordinate of the point.

        :returns: float -- X coordinate"""
        return self._x
    
    @property
    def y(self):
        """Retiurns Y coordinate of the point.

        :returns: float -- Y coordinate"""
        return self._y

    def draw(self, handle: DrawHandle):
        """Draw region to the image using the provided handle.

        :param handle: Draw handle
        :type handle: DrawHandle
        """
        handle.points([(self._x, self._y)])

    def resize(self, scale):
        """Resize region by the provided scale factor.

        :param scale: Scale factor
        :type scale: float
        """
        return Point(self._x * scale, self._y * scale)

    def is_empty(self):
        """Check if region is empty.

        Point regions are never empty by definition.
        """
        return False


from .raster import calculate_overlap, calculate_overlaps
from .shapes import Rectangle, Polygon, Mask

def is_special(region):
    """Check if the region is a special region.

    :param region: Region to check
    :type region: Region
    :returns: True if the region is a special region, False otherwise
    :rtype: bool"""
    
    return isinstance(region, Special)

def is_shape(region):
    """Check if the region is a shape region.

    :param region: Region to check
    :type region: Region
    :returns: True if the region is a shape region, False otherwise
    :rtype: bool"""
    return isinstance(region, (Rectangle, Polygon, Mask))