
import json
import os
import yaml
import collections
import datetime
import numpy as np

from vot.utilities.data import Grid

def open_utf8(path, mode='r'):
    """Open a file with UTF-8 encoding.

    :param path: Path to the file.
    :type path: str
    :param mode: Mode to open the file. Defaults to 'r'.
    :type mode: str, optional
    :returns: File object opened in the specified mode with UTF-8 encoding.
    :rtype: file object"""
    return open(path, mode, encoding='utf-8')

def touch(path, overwrite=True, content=""):
    """Create an empty file at the specified path.

    :param path: Path to the file to be created.
    :type path: str
    :param overwrite: If True, overwrite the file if it already exists. Defaults to True.
    :type overwrite: bool, optional
    :param content: Content to write to the file. Defaults to "".
    :type content: str, optional
    """
    if not overwrite and os.path.exists(path):
        return
    with open_utf8(path, 'w') as fp:
        os.utime(path, None)
        if content is not None:
            fp.write(content)



class JSONEncoder(json.JSONEncoder):
    """JSON encoder for internal types."""

    def default(self, o):
        """Default encoder."""
        if isinstance(o, Grid):
            return list(o)
        elif isinstance(o, datetime.date):
            return o.strftime('%Y/%m/%d')
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return super().default(o)

class YAMLEncoder(yaml.Dumper):
    """YAML encoder for internal types."""

    def represent_tuple(self, data):
        """Represents a tuple."""
        return self.represent_list(list(data))


    def represent_object(self, o):
        """Represents an object."""
        if isinstance(o, Grid):
            return self.represent_list(list(o))
        elif isinstance(o, datetime.date):
            return o.strftime('%Y/%m/%d')
        elif isinstance(o, np.ndarray):
            return self.represent_list(o.tolist())
        else:
            return super().represent_object(o)

YAMLEncoder.add_representer(collections.OrderedDict, YAMLEncoder.represent_dict)
YAMLEncoder.add_representer(tuple, YAMLEncoder.represent_tuple)
YAMLEncoder.add_representer(Grid, YAMLEncoder.represent_object)
YAMLEncoder.add_representer(np.ndarray,YAMLEncoder.represent_object)
YAMLEncoder.add_multi_representer(np.integer, YAMLEncoder.represent_int)
YAMLEncoder.add_multi_representer(np.inexact, YAMLEncoder.represent_float)