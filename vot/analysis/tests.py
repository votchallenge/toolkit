""" Unit tests for analysis module. """


import unittest

class Tests(unittest.TestCase):
    """ Unit tests for analysis module. """

    def test_perfect_accuracy(self):
        import numpy as np

        from vot.region import Rectangle, Special
        from vot.analysis.accuracy import gather_overlaps

        trajectory = [Rectangle(0, 0, 100, 100)] * 30
        groundtruth = [Rectangle(0, 0, 100, 100)] * 30

        trajectory[0] = Special(1)

        overlaps, _ = gather_overlaps(trajectory, groundtruth)

        print(overlaps)

        self.assertEqual(np.mean(overlaps), 1)