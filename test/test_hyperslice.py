import unittest
import json

from pysliceplorer import hyperslice_core as hs_core


class TestHyperSlice(unittest.TestCase):
    def f(self, x1, x2, x3, x4):
        return x1 + x2 + x3 + x4

    def test_min_max(self):
        with self.assertRaises(Exception):
            hs_core(self.f, 20, 10, 4, (0, 0, 0, 0), 10)

    def test_dim_consistency(self):
        with self.assertRaises(Exception):
            hs_core(self.f, 0, 10, 4, (0, 0, 0), 10)
        with self.assertRaises(Exception):
            hs_core(self.f, 0, 10, -5, (0, 0, 0, 0), 10)

    def test_result(self):
        out = hs_core(self.f, 1, 10, 4, (0, 0, 0, 0), 10)
        self.assertEqual(out.dim, 4, "Should report 4")
        self.assertEqual(len(out.x_grid), len(out.y_grid), "X grid and Y grid must have the same length")
        self.assertEqual(len(out.x_grid), 10, "Mesh grids length must be 10")
        with self.assertRaises(AttributeError):
            print(out.__plot)

    def test_data_fetch(self):
        out = hs_core(self.f, 1, 10, 4, (0, 0, 0, 0), 10)
        expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertListEqual(out.data(0, 0)[0], expected)
        self.assertListEqual(out.data(0, 0)[1], expected)
        self.assertListEqual(out.data(0, 0)[2], expected)
        with self.assertRaises(IndexError):
            print(out.data(0, 0)[11])
        with self.assertRaises(Exception):
            out.data(4, 6)
        with self.assertRaises(Exception):
            out.data(-4, 6)
        with self.assertRaises(Exception):
            out.data(1, 2.2)

    def test_json_fetch(self):
        out = hs_core(self.f, 1, 10, 4, (0, 0, 0, 0), 10)
        reload = json.loads(out.to_json())
        self.assertEqual(out.dim, reload['dim'])
        self.assertListEqual(list(out.x_grid[0]), reload['x_grid'][0])
        self.assertListEqual(list(out.y_grid[0]), reload['y_grid'][0])
        self.assertEqual(len(out.data(0, 0)), len(reload['entries']['0']['0']))
        self.assertListEqual(list(out.data(0, 0)[0]), reload['entries']['0']['0'][0])

if __name__ == '__main__':
    unittest.main()
