import os
import json
import glob

from abc import abstractmethod, ABC

class Stack(object):

    def __init__(self):
        pass

    @property
    def experiments(self):
        return []
        
    def __iter__(self):
        pass