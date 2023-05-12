import math
from typing import List, Union, TextIO
import struct
import io

import numpy as np
from numba import jit

@jit(nopython=True)
def mask_to_rle(m, maxstride=100000000):
    """
    # Input: 2-D numpy array
    # Output: list of numbers (1st number = #0s, 2nd number = #1s, 3rd number = #0s, ...)
    """
    # reshape mask to vector
    v = m.reshape((m.shape[0] * m.shape[1]))

    if v.size == 0:
        return [0]

    # output is empty at the beginning
    rle = []
    # index of the last different element
    last_idx = 0
    # check if first element is 1, so first element in RLE (number of zeros) must be set to 0
    if v[0] > 0:
        rle.append(0)

    # go over all elements and check if two consecutive are the same
    for i in range(1, v.size):
        if v[i] != v[i - 1]:
            length = i - last_idx
            # if length is larger than maxstride, split it into multiple elements
            while length > maxstride:
                rle.append(maxstride)
                rle.append(0)
                length -= maxstride
            # add remaining length
            if length > 0:
                rle.append(length)
            last_idx = i

    if v.size > 0:
        # handle last element of rle
        if last_idx < v.size - 1:
            # last element is the same as one element before it - add number of these last elements
            length = v.size - last_idx
            while length > maxstride:
                rle.append(maxstride)
                rle.append(0)
                length -= maxstride
            if length > 0:
                rle.append(length)
        else:
            # last element is different than one element before - add 1
            rle.append(1)

    return rle

@jit(nopython=True)
def rle_to_mask(rle, width, height):
    """
    rle: input rle mask encoding
    each evenly-indexed element represents number of consecutive 0s
    each oddly indexed element represents number of consecutive 1s
    width and height are dimensions of the mask
    output: 2-D binary mask
    """
    # allocate list of zeros
    v = [0] * (width * height)

    # set id of the last different element to the beginning of the vector
    idx_ = 0
    for i in range(len(rle)):
        if i % 2 != 0:
            # write as many 1s as RLE says (zeros are already in the vector)
            for j in range(rle[i]):
                v[idx_+j] = 1
        idx_ += rle[i]

    # reshape vector into 2-D mask
    # return np.reshape(np.array(v, dtype=np.uint8), (height, width)) # numba bug / not supporting np.reshape
    return np.array(v, dtype=np.uint8).reshape((height, width))

def create_mask_from_string(mask_encoding):
    """
    mask_encoding: a string in the following format: x0, y0, w, h, RLE
    output: mask, offset
    mask: 2-D binary mask, size defined in the mask encoding
    offset: (x, y) offset of the mask in the image coordinates
    """
    elements = [int(el) for el in mask_encoding]
    tl_x, tl_y, region_w, region_h = elements[:4]
    rle = np.array([el for el in elements[4:]], dtype=np.int32)

    # create mask from RLE within target region
    mask = rle_to_mask(rle, region_w, region_h)

    return mask, (tl_x, tl_y)

from vot.region.raster import mask_bounds

def encode_mask(mask):
    """
    mask: input binary mask, type: uint8
    output: full RLE encoding in the format: (x0, y0, w, h), RLE
    first get minimal axis-aligned region which contains all positive pixels
    extract this region from mask and calculate mask RLE within the region
    output position and size of the region, dimensions of the full mask and RLE encoding
    """
    # calculate coordinates of the top-left corner and region width and height (minimal region containing all 1s)
    x_min, y_min, x_max, y_max = mask_bounds(mask)

    # handle the case when the mask empty
    if x_min is None:
        return (0, 0, 0, 0), [0]
    else:
        tl_x = x_min
        tl_y = y_min
        region_w = x_max - x_min + 1
        region_h = y_max - y_min + 1

        # extract target region from the full mask and calculate RLE
        # do not use full mask to optimize speed and space
        target_mask = mask[tl_y:tl_y+region_h, tl_x:tl_x+region_w]
        rle = mask_to_rle(np.array(target_mask))

        return (tl_x, tl_y, region_w, region_h), rle

def parse_region(string: str) -> "Region":
    """   
        Parse input string to the appropriate region format and return Region object

    Args:
        string (str): comma separated list of values

    Returns:
        Region: resulting region
    """
    from vot.region import Special
    from vot.region.shapes import Rectangle, Polygon, Mask

    if string[0] == 'm':
        # input is a mask - decode it
        m_, offset_ = create_mask_from_string(string[1:].split(','))
        return Mask(m_, offset=offset_)
    else:
        # input is not a mask - check if special, rectangle or polygon
        tokens = [float(t) for t in string.split(',')]
        if len(tokens) == 1:
            return Special(tokens[0])
        if len(tokens) == 4:
            if any([math.isnan(el) for el in tokens]):
                return Special(0)
            else:
                return Rectangle(tokens[0], tokens[1], tokens[2], tokens[3])
        elif len(tokens) % 2 == 0 and len(tokens) > 4:
            if any([math.isnan(el) for el in tokens]):
                return Special(0)
            else:
                return Polygon([(x_, y_) for x_, y_ in zip(tokens[::2], tokens[1::2])])
    return None

def read_trajectory_binary(fp: io.RawIOBase):
    import struct
    from cachetools import LRUCache, cached
    from vot.region import Special
    from vot.region.shapes import Rectangle, Polygon, Mask

    buffer = dict(data=fp.read(), offset = 0)

    @cached(cache=LRUCache(maxsize=32))
    def calcsize(format):
        return struct.calcsize(format)

    def read(format: str):
        unpacked = struct.unpack_from(format, buffer["data"], buffer["offset"])
        buffer["offset"] += calcsize(format)
        return unpacked

    _, length = read("<hI")

    trajectory = []

    for _ in range(length):
        type, = read("<B")
        if type == 0: r = Special(*read("<I"))
        elif type == 1: r = Rectangle(*read("<ffff"))
        elif type == 2:
            n, = read("<H")
            values = read("<%df" % (2 * n))
            r = Polygon(list(zip(values[0::2], values[1::2])))
        elif type == 3:
            tl_x, tl_y, region_w, region_h, n = read("<hhHHH")
            rle = np.array(read("<%dH" % (n)), dtype=np.int32)
            r = Mask(rle_to_mask(rle, region_w, region_h), (tl_x, tl_y))
        else:
            raise IOError("Wrong region type")
        trajectory.append(r)
    return trajectory

def write_trajectory_binary(fp: io.RawIOBase, data: List["Region"]):
    import struct
    from vot.region import Special
    from vot.region.shapes import Rectangle, Polygon, Mask

    fp.write(struct.pack("<hI", 1, len(data)))

    for r in data:
        if isinstance(r, Special): fp.write(struct.pack("<BI", 0, r.code))
        elif isinstance(r, Rectangle): fp.write(struct.pack("<Bffff", 1, r.x, r.y, r.width, r.height))
        elif isinstance(r, Polygon): fp.write(struct.pack("<BH%df" % (2 * r.size), 2, r.size, *[item for sublist in r.points() for item in sublist]))
        elif isinstance(r, Mask): 
            rle = mask_to_rle(r.mask, maxstride=255*255)
            fp.write(struct.pack("<BhhHHH%dH" % len(rle), 3, r.offset[0], r.offset[1], r.mask.shape[1], r.mask.shape[0], len(rle), *rle))
        else:
            raise IOError("Wrong region type")

def read_trajectory(fp: Union[str, TextIO]):
    if isinstance(fp, str):
        try:
            import struct
            with open(fp, "r+b") as tfp:
                v, = struct.unpack("<h", tfp.read(struct.calcsize("<h")))
                binary = v == 1
                # TODO: we can use the same file handle in case of binary format
        except Exception as e:
            binary = False

        fp = open(fp, "rb" if binary else "r")
        close = True
    else:
        binary = isinstance(fp, (io.RawIOBase, io.BufferedIOBase)) 
        close = False

    if binary:
        regions = read_trajectory_binary(fp)
    else:
        regions = []
        for line in fp.readlines():
            regions.append(parse_region(line.strip()))

    if close:
        fp.close()

    return regions

def write_trajectory(fp: Union[str, TextIO], data: List["Region"]):
    """ Write a trajectory to a file handle or a file with a given name.

    Based on the suffix of a file or properties of a file handle, the output may be either text based
    or binary.

    Args:
        fp (Union[str, TextIO]): File handle or file name
        data (List[Region]): Trajectory, a list of region objects

    """

    if isinstance(fp, str):
        binary = fp.endswith(".bin")
        close = True
        fp = open(fp, "wb" if binary else "w")
    else:
        binary = isinstance(fp, (io.RawIOBase, io.BufferedIOBase)) 
        close = False

    if binary:
        write_trajectory_binary(fp, data)
    else:
        for region in data:
            fp.write(str(region) + "\n")
    
    if close:
        fp.close()