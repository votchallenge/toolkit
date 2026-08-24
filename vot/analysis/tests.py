"""Unit tests for analysis module."""


import unittest

class Tests(unittest.TestCase):
    """Unit tests for analysis module."""

    def test_perfect_accuracy(self):
        import numpy as np

        from vot.region import Rectangle, Special
        from vot.analysis.accuracy import gather_overlaps

        trajectory = [Rectangle(0, 0, 100, 100)] * 30
        groundtruth = [Rectangle(0, 0, 100, 100)] * 30

        trajectory[0] = Special(1)

        overlaps, _ = gather_overlaps(trajectory, groundtruth)

        self.assertEqual(np.mean(overlaps), 1)
    def test_vos_perfect_masks(self):
        import numpy as np

        from vot.analysis.vos import compute_f, compute_j, compute_jf
        from vot.region import Mask

        mask = Mask(np.ones((10, 10), dtype=np.uint8))

        self.assertEqual(compute_j([mask], [mask], (10, 10)), 1)
        self.assertEqual(compute_f([mask], [mask], (10, 10)), 1)
        self.assertEqual(compute_jf([mask], [mask], (10, 10)), 1)

    def test_vos_imperfect_mask(self):
        import numpy as np

        from vot.analysis.vos import compute_j, compute_jf
        from vot.region import Mask

        groundtruth = Mask(np.ones((10, 10), dtype=np.uint8))
        prediction = Mask(np.zeros((10, 10), dtype=np.uint8))

        self.assertEqual(compute_j([groundtruth], [prediction], (10, 10)), 0)
        self.assertLess(compute_jf([groundtruth], [prediction], (10, 10)), 1)

    def test_vos_input_validation(self):
        from vot.analysis.vos import compute_j

        with self.assertRaises(ValueError):
            compute_j([], [None])