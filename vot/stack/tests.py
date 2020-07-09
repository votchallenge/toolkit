import os
import unittest
import yaml

from vot.workspace import Workspace, VoidStorage
from vot.stack import Stack, list_integrated_stacks, resolve_stack

class NoWorkspace:

    @property
    def storage(self):
        return VoidStorage()

class TestStacks(unittest.TestCase):

    def test_stacks(self):
       
        stacks = list_integrated_stacks()
        for stack_name in stacks:
            try:
                with open(resolve_stack(stack_name), 'r') as fp:
                    stack_metadata = yaml.load(fp, Loader=yaml.BaseLoader)
                    Stack(stack_name, NoWorkspace(), **stack_metadata)
            except Exception as e:
                self.fail("Stack {}: {}".format(stack_name, e))