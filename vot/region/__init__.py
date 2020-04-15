from abc import abstractmethod, ABC
from typing import Tuple
from enum import Enum

from vot import VOTException
from vot.utilities.draw import DrawHandle

class RegionException(Exception):
    """General region exception"""

class ConversionException(RegionException):
    """Region conversion exception, the conversion cannot be performed
    """
    def __init__(self, *args, source=None):
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
    Base class for all region containers

    :var type: type of the region
    """
    def __init__(self):
        pass

    @property
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

    @abstractmethod
    def is_empty(self):
        """Check if region is empty (not annotated or not reported)
        """

class Special(Region):
    """
    Special region

    :var code: Code value
    """

    UNKNOWN = 0
    INITIALIZATION = 1
    FAILURE = 2

    def __init__(self, code):
        """ Constructor

        :param code: Special code
        """
        super().__init__()
        self._code = int(code)

    def __str__(self):
        """ Create string from class """
        return '{}'.format(self._code)

    @property
    def type(self):
        return RegionType.SPECIAL

    def copy(self):
        return Special(self._code)

    def convert(self, rtype: RegionType):
        if rtype == RegionType.SPECIAL:
            return self.copy()
        else:
            raise ConversionException("Unable to convert special region to {}".format(rtype))

    @property
    def code(self):
        """Retiurns special code for this region
        Returns:
            int -- Type code
        """
        return self._code

    def draw(self, handle: DrawHandle, color, width):
        pass

    def is_empty(self):
        return False

from vot.region.io import read_file, write_file
from .shapes import Rectangle, Polygon, Mask
from .io import read_file, write_file, parse
from .utils import calculate_overlap
