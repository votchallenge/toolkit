import math
from typing import Union, TextIO

import numpy as np
from numba import jit

from vot.region import Special

@jit(nopython=True)
def mask_to_rle(m):
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
            rle.append(i - last_idx)
            last_idx = i

    if v.size > 0:
        # handle last element of rle
        if last_idx < v.size - 1:
            # last element is the same as one element before it - add number of these last elements
            rle.append(v.size - last_idx)
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

def parse(string):
    """
    parse string to the appropriate region format and return region object
    """
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
    print('Unknown region format.')
    return None

def read_file(fp: Union[str, TextIO]):
    if isinstance(fp, str):
        with open(fp) as file:
            lines = file.readlines()
    else:
        lines = fp.readlines()

    regions = [0] * len(lines)
    # iterate over all lines in the file
    for i, line in enumerate(lines):
        regions[i] = parse(line.strip())
    return regions

def write_file(fp: Union[str, TextIO], data):
    """
    data is a list where each element is a region
    """
    if isinstance(fp, str):
        with open(fp, 'w') as file:
            for region in data:
                file.write('%s\n' % str(region))
    else:
        for region in data:
            fp.write('%s\n' % str(region))
