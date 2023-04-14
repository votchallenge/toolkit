
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

        r1 = Rectangle(0, 0, 0, 0)        
        self.assertEqual(calculate_overlap(r1, r1), 1)

    def test_empty_mask(self):
        from vot.region import Mask

        mask = Mask(np.zeros((100, 100), dtype=np.uint8))
        self.assertTrue(mask.is_empty())

        mask = Mask(np.ones((100, 100), dtype=np.uint8))
        self.assertFalse(mask.is_empty())

    def test_binary_format(self):
        """ Tests if the binary format of a region matched the plain-text one"""
        import io

        from vot.region import Rectangle, Polygon, Mask
        from vot.region.io import read_trajectory, write_trajectory
        from vot.region.raster import calculate_overlaps

        trajectory = [
            Rectangle(0, 0, 100, 100),
            Rectangle(0, 10, 100, 100),
            Rectangle(0, 0, 200, 100),
            Polygon([[0, 0], [0, 100], [100, 100], [100, 0]]),
            Mask(np.ones((100, 100), dtype=np.uint8)),
            Mask(np.zeros((100, 100), dtype=np.uint8)),
        ]

        binf = io.BytesIO()
        txtf = io.StringIO()

        write_trajectory(binf, trajectory)
        write_trajectory(txtf, trajectory)

        binf.seek(0)
        txtf.seek(0)

        bint = read_trajectory(binf)
        txtt = read_trajectory(txtf)

        o1 = calculate_overlaps(bint, txtt, None)
        o2 = calculate_overlaps(bint, trajectory, None)

        self.assertTrue(np.all(np.array(o1) == 1))
        self.assertTrue(np.all(np.array(o2) == 1))