""" This module contains classes for region representation and manipulation. Regions are also used to represent results
    of trackers as well as groundtruth trajectories. The module also contains functions for calculating overlaps between
    regions and for converting between different region types."""

from abc import abstractmethod, ABC
from enum import Enum

from vot import ToolkitException
from vot.utilities.draw import DrawHandle

class RegionException(ToolkitException):
    """General region exception"""

class ConversionException(RegionException):
    """Region conversion exception, the conversion cannot be performed
    """
    def __init__(self, *args, source=None):
        """Constructor

        Args:
            *args: Arguments for the base exception

        Keyword Arguments:
            source (Region): Source region (default: {None})

        """
        super().__init__(*args)
        self._source = source

class RegionType(Enum):
    """Enumeration of region types
    """
    SPECIAL = 0
    RECTANGLE = 1
    POLYGON = 2
    MASK = 3

class Region(ABC):
    """
    Base class for all region containers.
    """
    def __init__(self):
        """Base constructor"""
        pass

    @property
    @abstractmethod
    def type(self):
        """Return type of the region

        Returns:
            RegionType -- Type of the region
        """
        pass

    @abstractmethod
    def copy(self):
        """Copy region to another object

        Returns:
            Region -- Copy of the region
        """

    @abstractmethod
    def convert(self, rtype: RegionType):
        """Convert region to another type. Note that some conversions
        degrade information.
        
        Args:
            rtype (RegionType): Target region type to convert to.
        """

    @abstractmethod
    def is_empty(self):
        """Check if region is empty (not annotated or not reported)
        """

class Special(Region):
    """
    Special region, meaning of the code can change depending on the context

    :var code: Code value
    """

    def __init__(self, code):
        """ Constructor

        Args:
            code (int): Code value
        """
        super().__init__()
        self._code = int(code)

    def __str__(self):
        """ Create string from class """
        return '{}'.format(self._code)

    @property
    def type(self):
        """Return type of the region"""
        return RegionType.SPECIAL

    def copy(self):
        """Copy region to another object"""
        return Special(self._code)

    def convert(self, rtype: RegionType):
        """Convert region to another type. Note that some conversions degrade information.

        Args:
            rtype (RegionType): Target region type to convert to.

        Raises:
            ConversionException: Unable to convert special region to another type

        Returns:
            Region -- Converted region
        """

        if rtype == RegionType.SPECIAL:
            return self.copy()
        else:
            raise ConversionException("Unable to convert special region to {}".format(rtype))

    @property
    def code(self):
        """Retiurns special code for this region.
        Returns:
            int -- Type code
        """
        return self._code

    def draw(self, handle: DrawHandle):
        """Draw region to the image using the provided handle.

        Args:
            handle (DrawHandle): Draw handle
        """
        pass

    def is_empty(self):
        """ Check if region is empty. Special regions are always empty by definition."""
        return True

from .raster import calculate_overlap, calculate_overlaps
from .shapes import Rectangle, Polygon, Mask