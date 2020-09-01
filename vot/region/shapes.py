import sys

from copy import copy
from functools import reduce
from typing import Tuple, List
from abc import ABC, abstractmethod

import numpy as np
from numba import jit
import cv2

from vot.region import Region, ConversionException, RegionType, RegionException
from vot.utilities.draw import DrawHandle

class Shape(Region, ABC):

    @abstractmethod
    def draw(self, handle: DrawHandle) -> None:
        pass

    @abstractmethod
    def resize(self, factor=1) -> "Shape":
        pass

    @abstractmethod
    def move(self, dx=0, dy=0) -> "Shape":
        pass

    @abstractmethod
    def rasterize(self, bounds: Tuple[int, int, int, int]) -> np.ndarray:
        pass

    @abstractmethod
    def bounds(self) -> Tuple[int, int, int, int]:
        pass

class Rectangle(Shape):
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
        self._data = np.array([[x], [y], [width], [height]], dtype=np.float32)

    def __str__(self):
        """ Create string from class """
        return '{},{},{},{}'.format(self.x, self.y, self.width, self.height)

    @property
    def x(self):
        return self._data[0, 0]

    @property
    def y(self):
        return self._data[1, 0]

    @property
    def width(self):
        return self._data[2, 0]

    @property
    def height(self):
        return self._data[3, 0]

    @property
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
            points.append((self.x + self.width - 1, self.y))
            points.append((self.x + self.width - 1, self.y + self.height - 1))
            points.append((self.x, self.y + self.height - 1))
            return Polygon(points)
        elif rtype == RegionType.MASK:
            return Mask(np.ones((int(round(self.height)), int(round(self.width))), np.uint8), (int(round(self.x)), int(round(self.y))))
        else:
            raise ConversionException("Unable to convert rectangle region to {}".format(rtype), source=self)

    def is_empty(self):
        if self.width > 0 and self.height > 0:
            return False
        else:
            return True

    def draw(self, handle: DrawHandle):
        polygon = [(self.x, self.y), (self.x + self.width, self.y), \
            (self.x + self.width, self.y + self.height), \
            (self.x, self.y + self.height)]
        handle.polygon(polygon)

    def resize(self, factor=1):
        return Rectangle(self.x * factor, self.y * factor,
                         self.width * factor, self.height * factor)

    def center(self):
        return (self.x + self.width / 2, self.y + self.height / 2)

    def move(self, dx=0, dy=0):
        return Rectangle(self.x + dx, self.y + dy, self.width, self.height)

    def rasterize(self, bounds):
        from vot.region.raster import rasterize_rectangle
        return rasterize_rectangle(self._data, bounds)

    def bounds(self):
        return int(round(self.x)), int(round(self.y)), int(round(self.width + self.x)), int(round(self.height + self.y))

class Polygon(Shape):
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
        assert(points)
        self._points = np.array(points, dtype=np.float32)
        assert(self._points.shape[0] >= 3 and self._points.shape[1] == 2)  # pylint: disable=E1136


    def __str__(self):
        """ Create string from class """
        return ','.join(['{},{}'.format(p[0], p[1]) for p in self._points])

    @property
    def type(self):
        return RegionType.POLYGON

    @property
    def size(self):
        return self._points.shape[0] # pylint: disable=E1136

    def __getitem__(self, i):
        return self._points[i, 0], self._points[i, 1]

    def points(self):
        return [self[i] for i in range(self.size)]

    def copy(self):
        return copy(self)

    def convert(self, rtype: RegionType):
        if rtype == RegionType.POLYGON:
            return self.copy()
        elif rtype == RegionType.RECTANGLE:
            top = np.min(self._points[:, 1])
            bottom = np.max(self._points[:, 1])
            left = np.min(self._points[:, 0])
            right = np.max(self._points[:, 0])

            return Rectangle(left, top, right - left, bottom - top)
        elif rtype == RegionType.MASK:
            bounds = self.bounds()
            mask = self.rasterize(bounds)
            return Mask(mask, offset=(bounds[0], bounds[1]))
        else:
            raise ConversionException("Unable to convert polygon region to {}".format(rtype), source=self)

    def draw(self, handle: DrawHandle):
        handle.polygon([(p[0], p[1]) for p in self._points])

    def resize(self, factor=1):
        return Polygon([(p[0] * factor, p[1] * factor) for p in self._points])

    def move(self, dx=0, dy=0):
        return Polygon([(p[0] + dx, p[1] + dy) for p in self._points])

    def is_empty(self):
        top = np.min(self._points[:, 1])
        bottom = np.max(self._points[:, 1])
        left = np.min(self._points[:, 0])
        right = np.max(self._points[:, 0])
        return top == bottom or left == right

    def rasterize(self, bounds: Tuple[int, int, int, int]):
        from vot.region.raster import rasterize_polygon
        return rasterize_polygon(self._points, bounds)

    def bounds(self):
        top = np.min(self._points[:, 1])
        bottom = np.max(self._points[:, 1])
        left = np.min(self._points[:, 0])
        right = np.max(self._points[:, 0])
        return int(round(left)), int(round(top)), int(round(right)), int(round(bottom))

from vot.region.raster import mask_bounds
from vot.region.io import mask_to_rle

class Mask(Shape):
    """Mask region
    """

    def __init__(self, mask: np.array, offset: Tuple[int, int] = (0, 0), optimize=False):
        super().__init__()
        self._mask = mask.astype(np.uint8)
        self._mask[self._mask > 0] = 1
        self._offset = offset
        if optimize:  # optimize is used when mask without an offset is given (e.g. full-image mask)
            self._optimize()

    def __str__(self):
        offset_str = '%d,%d' % self.offset
        region_sz_str = '%d,%d' % (self.mask.shape[1], self.mask.shape[0])
        rle_str = ','.join([str(el) for el in mask_to_rle(self.mask)])
        return 'm%s,%s,%s' % (offset_str, region_sz_str, rle_str)

    def _optimize(self):
        bounds = mask_bounds(self.mask)
        if bounds[0] is None:
            # mask is empty
            self._mask = np.zeros((0, 0), dtype=np.uint8)
            self._offset = (0, 0)
        else:
            self._mask = np.copy(self.mask[bounds[1]:bounds[3], bounds[0]:bounds[2]])
            self._offset = (bounds[0] + self.offset[0], bounds[1] + self.offset[1])

    @property
    def mask(self):
        return self._mask

    @property
    def offset(self):
        return self._offset

    @property
    def type(self):
        return RegionType.MASK

    def copy(self):
        return copy(self)

    def convert(self, rtype: RegionType):
        if rtype == RegionType.MASK:
            return self.copy()
        elif rtype == RegionType.RECTANGLE:
            bounds = mask_bounds(self.mask)
            return Rectangle(bounds[0] + self.offset[0], bounds[1] + self.offset[1],
                            bounds[2] - bounds[0], bounds[3] - bounds[1])
        elif rtype == RegionType.POLYGON:
            bounds = mask_bounds(self.mask)
            if None in bounds:
                return Polygon([(0, 0), (0, 0), (0, 0), (0, 0)])
            return Polygon([
                (bounds[0] + self.offset[0], bounds[1] + self.offset[1]),
                (bounds[2] + self.offset[0], bounds[1] + self.offset[1]),
                (bounds[2] + self.offset[0], bounds[3] + self.offset[1]),
                (bounds[0] + self.offset[0], bounds[3] + self.offset[1])])
        else:
            raise ConversionException("Unable to convert mask region to {}".format(rtype), source=self)

    def draw(self, handle: DrawHandle):
        handle.mask(self._mask, self.offset)

    def rasterize(self, bounds):
        from vot.region.raster import copy_mask
        return copy_mask(self._mask, self._offset, bounds)

    def is_empty(self):
        if self.mask.shape[1] > 0 and self.mask.shape[0] > 0:
            return False
        else:
            return True

    def resize(self, factor=1):

        offset = (int(self.offset[0] * factor), int(self.offset[1] * factor))
        height = max(1, int(self.mask.shape[0] * factor))
        width = max(1, int(self.mask.shape[1] * factor))

        if self.mask.size == 0:
            mask = np.zeros((0, 0), dtype=np.uint8)
        else:
            mask = cv2.resize(self.mask, dsize=(width, height), interpolation=cv2.INTER_NEAREST)

        return Mask(mask, offset, False)

    def move(self, dx=0, dy=0):
        return Mask(self._mask, (self.offset[0] + dx, self.offset[1] + dy))

    def bounds(self):
        bounds = mask_bounds(self.mask)
        return bounds[0] + self.offset[0], bounds[1] + self.offset[1], bounds[2] + self.offset[0], bounds[3] + self.offset[1]
