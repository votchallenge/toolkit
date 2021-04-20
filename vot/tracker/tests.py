

import unittest

from ..dataset.dummy import DummySequence
from ..tracker.dummy import DummyTracker

class TestStacks(unittest.TestCase):

    def test_tracker_test(self):
       
        tracker = DummyTracker
        sequence = DummySequence(10)

        with tracker.runtime(log=False) as runtime:
            runtime.initialize(sequence.frame(0), sequence.groundtruth(0))
            for i in range(1, sequence.length):
                runtime.update(sequence.frame(i))
