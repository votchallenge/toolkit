import math
from typing import Union, TextIO

from vot.region import Special
from vot.region.shapes import Rectangle, Polygon, Mask
from vot.region.utils import create_mask_from_string

def parse(string):
    """
    parse string to the appropriate region format and return region object
    """
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
