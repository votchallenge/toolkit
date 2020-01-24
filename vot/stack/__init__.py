import os
import json
import glob
from typing import List

from abc import abstractmethod, ABC

from vot.experiment import Experiment
from vot.utilities import import_class

class Stack(object):

    def __init__(self, *args: List[Experiment]):
        self._experiments = list(args)

    @property
    def experiments(self) -> List[Experiment]:
        return self._experiments
        
    def __iter__(self):
        return iter(self._experiments)

from vot.stack.challenges import VOT2013, VOT2014, VOT2015, VOT2016, VOT2017, VOT2018, VOT2019

_stacks = dict(vot2013=VOT2013(), \
    vot2014=VOT2014(), vot2015=VOT2015(), vot2016=VOT2016(), \
    vot2017=VOT2017(), vot2018=VOT2018(), vot2019=VOT2019())

def resolve_stack(name):
    if name in _stacks:
        return _stacks[name]
    cls = import_class(name)
    assert issubclass(cls, Stack)  # [AL] Check if this is ok
    return cls()