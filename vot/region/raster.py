from typing import List, Tuple, Optional

from numba import njit
import numpy as np

from vot.region import Region
from vot.region.shapes import Shape

@njit
def mask_overlap(m1, m2):
    a1 = m1.ravel()
    a2 = m2.ravel()

    intersection = 0
    union = 0

    for i in range(a1.size):
        if a1[i] != 0 and a2[i] != 0:
            intersection += 1
        if a1[i] != 0 or a2[i] != 0:
            union += 1

    return float(intersection) / float(union) if union > 0 else float(0)

def calculate_overlap(reg1: Shape, reg2: Shape, bounds: Optional[Tuple[int, int]] = None):
    """
    Inputs: reg1 and reg2 are Region objects (Rectangle, Polygon or Mask)
    bounds: size of the image, format: [width, height]
    function first rasterizes both regions to 2-D binary masks and calculates overlap between them
    """

    if not isinstance(reg1, Shape) or not isinstance(reg2, Shape):
        return float(0)

    bounds1 = reg1.bounds()
    bounds2 = reg2.bounds()

    union = (min(bounds1[0], bounds2[0]), min(bounds1[1], bounds2[1]), max(bounds1[2], bounds2[2]), max(bounds1[3], bounds2[3]))

    if not bounds is None:
        bounds = (min(0, union[0]), min(0, union[1]), max(bounds[0], union[2]), max(bounds[1], union[3]))
    else:
        bounds = union

    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        return float(0)

    # convert both regions to mask
    m1 = reg1.rasterize(bounds)
    m2 = reg2.rasterize(bounds)

    return mask_overlap(m1, m2)

def calculate_overlaps(first: List[Region], second: List[Region], bounds: Optional[Tuple[int, int]]):
    """
    first and second are lists containing objects of type Region
    bounds is in the format [width, height]
    output: list of per-frame overlaps (floats)
    """
    assert(len(first) == len(second))
    return [calculate_overlap(pairs[0], pairs[1], bounds=bounds) for i, pairs in enumerate(zip(first, second))]

@njit
def rasterize_rectangle(x, y, width, height, bounds: Tuple[int, int, int, int]):
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]

    mask = np.zeros((height, width), dtype=np.uint8)

    if x > bounds[2] or x + width < bounds[0] or y > bounds[3] or y + height < bounds[1]:
        return mask

    left = max(0, x - bounds[0])
    top = max(0, y - bounds[1])
    right = min(bounds[2] - bounds[0], width)
    bottom = min(bounds[3] - bounds[1], height)

    mask[top:bottom, left:right] = 1

    return mask


@njit
def rasterize_polygon(data: np.ndarray, bounds: Tuple[int, int, int, int]):

    #int nodes, pixelY, i, j, swap;
    #region_polygon polygon = polygon_input;
    count = data.shape[0]

    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]

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


@njit
def copy_mask(mask: np.ndarray, offset: Tuple[int, int], bounds):
    tx = max(offset[0], bounds[0])
    ty = max(offset[1], bounds[1])

    ox = tx - bounds[0]
    oy = ty - bounds[1]
    gx = tx - offset[0]
    gy = ty - offset[1]

    tw = min(bounds[0] + bounds[2], offset[0] + mask.shape[1]) - tx
    th = min(bounds[1] + bounds[3], offset[1] + mask.shape[0]) - ty

    copy = np.zeros((bounds[3], bounds[2]), dtype=np.uint8)

    #copy[]

    for i in range(th):
        for j in range(tw):
            copy[i + oy, j + ox] = mask[i + gy, j + gx]

    return mask