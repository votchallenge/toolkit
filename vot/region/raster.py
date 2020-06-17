from typing import List, Tuple, Optional

import numba
import numpy as np

#import llvmlite.binding as llvm
#llvm.set_option('', '--debug-only=loop-vectorize')

@numba.njit(inline='always')
def mask_bounds(mask: np.ndarray):
    """
    mask: 2-D array with a binary mask
    output: coordinates of the top-left and bottom-right corners of the minimal axis-aligned region containing all positive pixels
    """
    ii32 = np.iinfo(np.int32)
    top = ii32.max
    bottom = ii32.min
    left = ii32.max
    right = ii32.min

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] != 0:
                top = min(top, i)
                bottom = max(bottom, i)
                left = min(left, j)
                right = max(right, j)

    return (left, top, right, bottom)


@numba.njit(inline='always')
def rasterize_rectangle(data: np.ndarray, bounds: Tuple[int, int, int, int]):
    width = bounds[2] - bounds[0] + 1
    height = bounds[3] - bounds[1] + 1

    mask = np.zeros((height, width), dtype=np.uint8)

    if data[0, 0] > bounds[2] or data[0, 0] + data[2, 0] - 1 < bounds[0] or data[1, 0] > bounds[3] or data[1, 0] + data[3, 0] - 1 < bounds[1]:
        return mask

    left = max(0, data[0, 0] - bounds[0])
    top = max(0, data[1, 0] - bounds[1])
    right = min(bounds[2], data[0, 0] + data[2, 0] - 1 - bounds[0])
    bottom = min(bounds[3], data[1, 0] + data[3, 0] - 1 - bounds[1])

    mask[top:bottom+1, left:right+1] = 1

    return mask


@numba.njit(numba.uint8[:, ::1](numba.float32[:, ::1], numba.types.UniTuple(numba.int64, 4)), inline='always')
def rasterize_polygon(data: np.ndarray, bounds: Tuple[int, int, int, int]):

    #int nodes, pixelY, i, j, swap;
    #region_polygon polygon = polygon_input;
    count = data.shape[0]

    width = bounds[2] - bounds[0] + 1
    height = bounds[3] - bounds[1] + 1

    nodeX = np.zeros((count, ), dtype=np.int64)
    mask = np.zeros((height, width), dtype=np.uint8)

    polygon = np.empty_like(data)
    np.round(data, 0, polygon)

    polygon = polygon - np.array([[bounds[0], bounds[1]]])

    #  Loop through the rows of the image.
    for pixelY in range(height):

        #  Build a list of nodes.
        nodes = 0
        j = count - 1

        for i in range(count):
            if (((polygon[i, 1] <= pixelY) and (polygon[j, 1] > pixelY)) or
                    ((polygon[j, 1] <= pixelY) and (polygon[i, 1] > pixelY)) or
                    ((polygon[i, 1] < pixelY) and (polygon[j, 1] >= pixelY)) or
                    ((polygon[j, 1] < pixelY) and (polygon[i, 1] >= pixelY)) or
                    ((polygon[i, 1] == polygon[j, 1]) and (polygon[i, 1] == pixelY))):
                r = (polygon[j, 1] - polygon[i, 1])
                k = (polygon[j, 0] - polygon[i, 0])
                if r != 0:
                    nodeX[nodes] = (polygon[i, 0] + (pixelY - polygon[i, 1]) / r * k)
                else:
                    nodeX[nodes] = polygon[i, 0]
                nodes = nodes + 1
            j = i

        # Sort the nodes, via a simple “Bubble” sort.
        i = 0
        while (i < nodes - 1):
            if nodeX[i] > nodeX[i + 1]:
                swap = nodeX[i]
                nodeX[i] = nodeX[i + 1]
                nodeX[i + 1] = swap
                if (i):
                    i = i - 1
            else:
                i = i + 1

        #  Fill the pixels between node pairs.
        i = 0
        while i < nodes - 1:
            if nodeX[i] >= width:
                break
            # If a point is in the line then we get two identical values
            # Ignore the first, except when it is the last point in vector
            if (nodeX[i] == nodeX[i + 1] and i < nodes - 2):
                i = i + 1
                continue

            if nodeX[i + 1] >= 0:
                if nodeX[i] < 0:
                    nodeX[i] = 0
                if nodeX[i + 1] >= width:
                    nodeX[i + 1] = width - 1
                for j in range(nodeX[i], nodeX[i + 1] + 1):
                    mask[pixelY, j] = 1
            i += 2

    return mask


@numba.njit(inline='always')
def copy_mask(mask: np.ndarray, offset: Tuple[int, int], bounds: Tuple[int, int, int, int]):
    tx = max(offset[0], bounds[0])
    ty = max(offset[1], bounds[1])

    ox = tx - bounds[0]
    oy = ty - bounds[1]
    gx = tx - offset[0]
    gy = ty - offset[1]

    tw = min(bounds[2] + 1, offset[0] + mask.shape[1]) - tx
    th = min(bounds[3] + 1, offset[1] + mask.shape[0]) - ty

    copy = np.zeros((bounds[3] - bounds[1] + 1, bounds[2] - bounds[0] + 1), dtype=np.uint8)

    for i in range(th):
        for j in range(tw):
            copy[i + oy, j + ox] = mask[i + gy, j + gx]

    return copy

@numba.njit(inline='always')
def _bounds_rectangle(a):
    return (int(round(a[0, 0])), int(round(a[1, 0])), int(round(a[0, 0] + a[2, 0] - 1)), int(round(a[1, 0] + a[3, 0] - 1)))

@numba.njit(inline='always')
def _bounds_polygon(a):
    fi32 = np.finfo(np.float32)
    top = fi32.max
    bottom = fi32.min
    left = fi32.max
    right = fi32.min

    for i in range(a.shape[0]):
        top = min(top, a[i, 1])
        bottom = max(bottom, a[i, 1])
        left = min(left, a[i, 0])
        right = max(right, a[i, 0])
    return (int(round(left)), int(round(top)), int(round(right)), int(round(bottom)))

@numba.njit(inline='always')
def _bounds_mask(a, o):
    bounds = mask_bounds(a)
    return (bounds[0] + o[0], bounds[1] + o[1], bounds[2] + o[0], bounds[3] + o[1])

@numba.njit(inline='always')
def _region_bounds(a, o):
    if a.shape[0] == 4  and a.shape[1] == 1:
        return _bounds_rectangle(a)
    elif a.shape[0] > 3 and a.shape[1] == 2:
        return _bounds_polygon(a)
    elif not o is None:
        return _bounds_mask(a, o)
    return (0, 0, 0, 0)

@numba.njit(inline='always')
def _region_raster(a: np.ndarray, bounds: Tuple[int, int, int, int], o: Optional[Tuple[int, int]] = None):

    if a.shape[0] == 4  and a.shape[1] == 1:
        return rasterize_rectangle(a, bounds)
    elif a.shape[0] > 3 and a.shape[1] == 2:
        return rasterize_polygon(a, bounds)
    elif not o is None:
        return copy_mask(a, o, bounds)

@numba.njit(inline='always', cache=True)
def _calculate_overlap(a: np.ndarray, b: np.ndarray, ao: Optional[Tuple[int, int]] = None,
        bo: Optional[Tuple[int, int]] = None, bounds: Optional[Tuple[int, int]] = None):

    bounds1 = _region_bounds(a, ao)
    bounds2 = _region_bounds(b, bo)

    union = (min(bounds1[0], bounds2[0]), min(bounds1[1], bounds2[1]), max(bounds1[2], bounds2[2]), max(bounds1[3], bounds2[3]))

    if not bounds is None:
        raster_bounds = (max(0, union[0]), max(0, union[1]), min(bounds[0] - 1, union[2]), min(bounds[1] - 1, union[3]))
    else:
        raster_bounds = union

    if raster_bounds[0] >= raster_bounds[2] or raster_bounds[1] >= raster_bounds[3]:
        return float(0)

    m1 = _region_raster(a, raster_bounds, ao)
    m2 = _region_raster(b, raster_bounds, bo)

    a1 = m1.ravel()
    a2 = m2.ravel()

    intersection = 0
    union_ = 0

    for i in range(a1.size):
        if a1[i] != 0 or a2[i] != 0:
            union_ += 1
            if a1[i] != 0 and a2[i] != 0:
                intersection += 1

    return float(intersection) / float(union_) if union_ > 0 else float(0)

from vot.region import Region, RegionException
from vot.region.shapes import Shape, Rectangle, Polygon, Mask

def calculate_overlap(reg1: Shape, reg2: Shape, bounds: Optional[Tuple[int, int]] = None):
    """
    Inputs: reg1 and reg2 are Region objects (Rectangle, Polygon or Mask)
    bounds: size of the image, format: [width, height]
    function first rasterizes both regions to 2-D binary masks and calculates overlap between them
    """

    if not isinstance(reg1, Shape) or not isinstance(reg2, Shape):
        return float(0)

    if isinstance(reg1, Rectangle):
        data1 = np.round(reg1._data)
        offset1 = None
    elif isinstance(reg1, Polygon):
        data1 = np.round(reg1._points)
        offset1 = None
    elif isinstance(reg1, Mask):
        data1 = reg1.mask
        offset1 = reg1.offset

    if isinstance(reg2, Rectangle):
        data2 = np.round(reg2._data)
        offset2 = None
    elif isinstance(reg2, Polygon):
        data2 = np.round(reg2._points)
        offset2 = None
    elif isinstance(reg2, Mask):
        data2 = reg2.mask
        offset2 = reg2.offset

    return _calculate_overlap(data1, data2, offset1, offset2, bounds)

def calculate_overlaps(first: List[Region], second: List[Region], bounds: Optional[Tuple[int, int]]):
    """
    first and second are lists containing objects of type Region
    bounds is in the format [width, height]
    output: list of per-frame overlaps (floats)
    """
    if not len(first) == len(second):
        raise RegionException("List not of the same size {} != {}".format(len(first), len(second)))
    return [calculate_overlap(pairs[0], pairs[1], bounds=bounds) for i, pairs in enumerate(zip(first, second))]
