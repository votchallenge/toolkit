""" Module for region shapes. """

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
    """ Base class for all shape regions. """

    @abstractmethod
    def draw(self, handle: DrawHandle) -> None:
        """ Draw the region to the given handle. 
 
        """
        pass

    @abstractmethod
    def resize(self, factor=1) -> "Shape":
        """ Resize the region by the given factor. """
        pass

    @abstractmethod
    def move(self, dx=0, dy=0) -> "Shape":
        """ Move the region by the given offset. 
        
        Args:
            dx (float, optional): X offset. Defaults to 0.
            dy (float, optional): Y offset. Defaults to 0.
            
        Returns:
            Shape: Moved region.
        """
        pass

    @abstractmethod
    def rasterize(self, bounds: Tuple[int, int, int, int]) -> np.ndarray:
        """ Rasterize the region to a binary mask.

        Args:
            bounds (Tuple[int, int, int, int]): Bounds of the mask.

        Returns:
            np.ndarray: Binary mask.
        """ 
        pass

    @abstractmethod
    def bounds(self) -> Tuple[int, int, int, int]:
        """ Get the bounding box of the region.
        
        Returns:
            Tuple[int, int, int, int]: Bounding box (x, y, width, height).
        """

        pass

class Rectangle(Shape):
    """
    Rectangle region class for representing rectangular regions.
    """
    def __init__(self, x=0, y=0, width=0, height=0):
        """ Constructor for rectangle region.

        Args:
            x (float, optional): X coordinate of the top left corner. Defaults to 0.
            y (float, optional): Y coordinate of the top left corner. Defaults to 0.
            width (float, optional): Width of the rectangle. Defaults to 0.
            height (float, optional): Height of the rectangle. Defaults to 0.
        """
        super().__init__()
        self._data = np.array([[x], [y], [width], [height]], dtype=np.float32)

    def __str__(self):
        """ Create string from class """
        return '{},{},{},{}'.format(self.x, self.y, self.width, self.height)

    @property
    def x(self):
        """ X coordinate of the top left corner. """
        return float(self._data[0, 0])

    @property
    def y(self):
        """ Y coordinate of the top left corner. """
        return float(self._data[1, 0])

    @property
    def width(self):
        """ Width of the rectangle."""
        return float(self._data[2, 0])

    @property
    def height(self):
        """ Height of the rectangle."""
        return float(self._data[3, 0])

    @property
    def type(self):
        """ Type of the region."""
        return RegionType.RECTANGLE

    def copy(self):
        """ Copy region to another object. """
        return copy(self)

    def convert(self, rtype: RegionType):
        """ Convert region to another type. Note that some conversions degrade information.
        
        Args:
            rtype (RegionType): Desired type.
        
        Raises:
            ConversionException: Unable to convert rectangle region to {rtype}
        """
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
        """ Check if the region is empty.
        
        Returns:
            bool: True if the region is empty, False otherwise.
        """
        if self.width > 0 and self.height > 0:
            return False
        else:
            return True

    def draw(self, handle: DrawHandle):
        """ Draw the region to the given handle.
        
        Args:
            handle (DrawHandle): Handle to draw to.
        """
        polygon = [(self.x, self.y), (self.x + self.width, self.y), \
            (self.x + self.width, self.y + self.height), \
            (self.x, self.y + self.height)]
        handle.polygon(polygon)

    def resize(self, factor=1):
        """ Resize the region by the given factor.
        
        Args:
            factor (float, optional): Resize factor. Defaults to 1.
            
        Returns:
            Rectangle: Resized region.
        """
        return Rectangle(self.x * factor, self.y * factor,
                         self.width * factor, self.height * factor)

    def center(self):
        """ Get the center of the region.
        
        Returns:
            tuple: Center coordinates (x,y).
        """
        return (self.x + self.width / 2, self.y + self.height / 2)

    def move(self, dx=0, dy=0):
        """ Move the region by the given offset.
        
        Args:
            dx (float, optional): X offset. Defaults to 0.
            dy (float, optional): Y offset. Defaults to 0.
            
        Returns:
            Rectangle: Moved region.
        """
        return Rectangle(self.x + dx, self.y + dy, self.width, self.height)

    def rasterize(self, bounds: Tuple[int, int, int, int]):
        """ Rasterize the region to a binary mask.
        
        Args:
            bounds (tuple): Bounds of the mask (x1,y1,x2,y2).
        """
        from vot.region.raster import rasterize_rectangle
        return rasterize_rectangle(self._data, np.array(bounds))

    def bounds(self):
        """ Get the bounding box of the region.
        
        Returns:
            tuple: Bounding box (x1,y1,x2,y2).
        """
        return int(round(self.x)), int(round(self.y)), int(round(self.width + self.x)), int(round(self.height + self.y))

class Polygon(Shape):
    """
    Polygon region defined by a list of points. The polygon is closed, i.e. the first and last point are connected.
    """
    def __init__(self, points):
        """
        Constructor

        Args:
            points (list): List of points as tuples [(x1,y1), (x2,y2),...,(xN,yN)]
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
        """ Get the region type. """
        return RegionType.POLYGON

    @property
    def size(self):
        """ Get the number of points. """
        return self._points.shape[0] # pylint: disable=E1136

    def __getitem__(self, i):
        """ Get the i-th point."""
        return self._points[i, 0], self._points[i, 1]

    def points(self):
        """ Get the list of points. 
        
        Returns:
            list: List of points as tuples [(x1,y1), (x2,y2),...,(xN,yN)]
        """
        return [self[i] for i in range(self.size)]

    def copy(self):
        """ Create a copy of the polygon. """
        return copy(self)

    def convert(self, rtype: RegionType):
        """ Convert the polygon to another region type.
        
        Args:
            rtype (RegionType): Target region type.
            
        Returns:
            Region: Converted region.
        """
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
        """ Draw the polygon on the given handle.
        
        Args:
            handle (DrawHandle): Handle to draw on.
        """
        handle.polygon([(p[0], p[1]) for p in self._points])

    def resize(self, factor=1):
        """ Resize the polygon by a factor.
        
        Args:
            factor (float): Resize factor.
            
        Returns:
            Polygon: Resized polygon.
        """
        return Polygon([(p[0] * factor, p[1] * factor) for p in self._points])

    def move(self, dx=0, dy=0):
        """ Move the polygon by a given offset.
        
        Args:
            dx (float): X offset.
            dy (float): Y offset.
            
        Returns:
            Polygon: Moved polygon.
        """
        return Polygon([(p[0] + dx, p[1] + dy) for p in self._points])

    def is_empty(self):
        """ Check if the polygon is empty.
        
        Returns:
            bool: True if the polygon is empty, False otherwise.
            
        """
        top = np.min(self._points[:, 1])
        bottom = np.max(self._points[:, 1])
        left = np.min(self._points[:, 0])
        right = np.max(self._points[:, 0])
        return top == bottom or left == right

    def rasterize(self, bounds: Tuple[int, int, int, int]):
        """ Rasterize the polygon into a binary mask.
        
        Args:
            bounds (tuple): Bounding box of the mask as (left, top, right, bottom).
            
        Returns:
            numpy.ndarray: Binary mask.
        """
        from vot.region.raster import rasterize_polygon
        return rasterize_polygon(self._points, bounds)

    def bounds(self):
        """ Get the bounding box of the polygon.

        Returns:
            tuple: Bounding box as (left, top, right, bottom).
        """
        top = np.min(self._points[:, 1])
        bottom = np.max(self._points[:, 1])
        left = np.min(self._points[:, 0])
        right = np.max(self._points[:, 0])
        return int(round(left)), int(round(top)), int(round(right)), int(round(bottom))

from vot.region.raster import mask_bounds
from vot.region.io import mask_to_rle

class Mask(Shape):
    """Mask region defined by a binary mask. The mask is defined by a binary image and an offset.
    """

    def __init__(self, mask: np.array, offset: Tuple[int, int] = (0, 0), optimize=False):
        """ Constructor
        
        Args:
            mask (numpy.ndarray): Binary mask.
            offset (tuple): Offset of the mask as (x, y). 
            optimize (bool): Optimize the mask by removing empty rows and columns.
            
        """
        super().__init__()
        self._mask = mask.astype(np.uint8)
        self._mask[self._mask > 0] = 1
        self._offset = offset
        if optimize:  # optimize is used when mask without an offset is given (e.g. full-image mask)
            self._optimize()
            
    def __str__(self):
        """ Create string from class """
        offset_str = '%d,%d' % self.offset
        region_sz_str = '%d,%d' % (self.mask.shape[1], self.mask.shape[0])
        rle_str = ','.join([str(el) for el in mask_to_rle(self.mask)])
        return 'm%s,%s,%s' % (offset_str, region_sz_str, rle_str)

    def _optimize(self):
        """ Optimize the mask by removing empty rows and columns. If the mask is empty, the mask is set to zero size.
         Do not call this method directly, it is called from the constructor. """
        bounds = mask_bounds(self.mask)
        if bounds[2] == 0:
            # mask is empty
            self._mask = np.zeros((0, 0), dtype=np.uint8)
            self._offset = (0, 0)
        else:

            self._mask = np.copy(self.mask[bounds[1]:bounds[3]+1, bounds[0]:bounds[2]+1])
            self._offset = (bounds[0] + self.offset[0], bounds[1] + self.offset[1])

    @property
    def mask(self):
        """ Get the mask. Note that you should not modify the mask directly. Also make sure to
        take into account the offset when using the mask."""
        return self._mask

    @property
    def offset(self):
        """ Get the offset of the mask in pixels."""
        return self._offset

    @property
    def type(self):
        """ Get the region type."""
        return RegionType.MASK

    def copy(self):
        """ Create a copy of the mask."""
        return copy(self)

    def convert(self, rtype: RegionType):
        """ Convert the mask to another region type. The mask is converted to a rectangle or polygon by approximating bounding box of the mask.
        
        Args:
            rtype (RegionType): Target region type.
            
        Returns:
            Shape: Converted region.
        """
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
        """ Draw the mask into an image.
        
        Args:
            handle (DrawHandle): Handle to the image.
        """
        handle.mask(self._mask, self.offset)

    def rasterize(self, bounds: Tuple[int, int, int, int]):
        """ Rasterize the mask into a binary mask. The mask is cropped to the given bounds.
        
        Args:
            bounds (tuple): Bounding box of the mask as (left, top, right, bottom).

        Returns:
            numpy.ndarray: Binary mask. The mask is a copy of the original mask.
        """
        from vot.region.raster import copy_mask
        return copy_mask(self._mask, self._offset, np.array(bounds))

    def is_empty(self):
        """ Check if the mask is empty.
        
        Returns:
            bool: True if the mask is empty, False otherwise.
        """
        bounds = mask_bounds(self.mask)
        return bounds[2] == 0 or bounds[3] == 0

    def resize(self, factor=1):
        """ Resize the mask by a given factor. The mask is resized using nearest neighbor interpolation.
        
        Args:
            factor (float): Resize factor.
            
        Returns:
            Mask: Resized mask.
        """

        offset = (int(self.offset[0] * factor), int(self.offset[1] * factor))
        height = max(1, int(self.mask.shape[0] * factor))
        width = max(1, int(self.mask.shape[1] * factor))

        if self.mask.size == 0:
            mask = np.zeros((0, 0), dtype=np.uint8)
        else:
            mask = cv2.resize(self.mask, dsize=(width, height), interpolation=cv2.INTER_NEAREST)

        return Mask(mask, offset, False)

    def move(self, dx=0, dy=0):
        """ Move the mask by a given offset.

        Args:
            dx (int): Horizontal offset.
            dy (int): Vertical offset.

        Returns:
            Mask: Moved mask.
        """
        return Mask(self._mask, (self.offset[0] + dx, self.offset[1] + dy))

    def bounds(self):
        """ Get the bounding box of the mask.

        Returns:
            tuple: Bounding box of the mask as (left, top, right, bottom).
        """
        bounds = mask_bounds(self.mask)
        return bounds[0] + self.offset[0], bounds[1] + self.offset[1], bounds[2] + self.offset[0], bounds[3] + self.offset[1]
