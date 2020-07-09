
import unittest

import numpy as np

from vot.region.raster import rasterize_polygon, rasterize_rectangle, copy_mask, calculate_overlap

class TestRasterMethods(unittest.TestCase):

    def test_rasterize_polygon(self):
        points = np.array([[0, 0], [0, 100], [100, 100], [100, 0]], dtype=np.float32)
        np.testing.assert_array_equal(rasterize_polygon(points, (0, 0, 99, 99)), np.ones((100, 100), dtype=np.uint8))

    def test_rasterize_rectangle(self):
        np.testing.assert_array_equal(rasterize_rectangle(np.array([[0], [0], [100], [100]], dtype=np.float32), (0, 0, 99, 99)), np.ones((100, 100), dtype=np.uint8))

    def test_copy_mask(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        np.testing.assert_array_equal(copy_mask(mask, (0, 0), (0, 0, 99, 99)), np.ones((100, 100), dtype=np.uint8))

    def test_calculate_overlap(self):
        from vot.region import Rectangle

        r1 = Rectangle(0, 0, 100, 100)
        self.assertEqual(calculate_overlap(r1, r1), 1)

