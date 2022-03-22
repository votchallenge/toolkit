
import unittest
import yaml

from vot.workspace import NullStorage
from vot.stack import Stack, list_integrated_stacks, resolve_stack

class NoWorkspace:
    """Empty workspace, does not save anything
    """

    @property
    def storage(self):
        return NullStorage()

class TestStacks(unittest.TestCase):
    """Tests for the experiment stack utilities
    """

    def test_stacks(self):
        """Test loading integrated stacks
        """
       
        stacks = list_integrated_stacks()
        for stack_name in stacks:
            try:
                with open(resolve_stack(stack_name), 'r') as fp:
                    stack_metadata = yaml.load(fp, Loader=yaml.BaseLoader)
                    Stack(stack_name, NoWorkspace(), **stack_metadata)
            except Exception as e:
                self.fail("Stack {}: {}".format(stack_name, e))