
import json
import yaml
import collections
import datetime
import numpy as np

from vot.utilities.data import Grid

class JSONEncoder(json.JSONEncoder):
    """ JSON encoder for internal types. """

    def default(self, o):
        """ Default encoder. """
        if isinstance(o, Grid):
            return list(o)
        elif isinstance(o, datetime.date):
            return o.strftime('%Y/%m/%d')
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return super().default(o)

class YAMLEncoder(yaml.Dumper):
    """ YAML encoder for internal types."""

    def represent_tuple(self, data):
        """ Represents a tuple. """
        return self.represent_list(list(data))


    def represent_object(self, o):
        """ Represents an object. """
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