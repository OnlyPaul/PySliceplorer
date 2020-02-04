import unittest
import json

from pysliceplorer import sliceplorer_core as sp_core


class TestSliceplorer(unittest.TestCase):
    def f(self, x1, x2, x3, x4):
        return x1 + x2 + x3 + x4

    def test_min_max(self):
        with self.assertRaises(Exception):
            sp_core(self.f, 20, 10, 4, 5, 10)

    def test_dim_consistency(self):
        with self.assertRaises(Exception):
            sp_core(self.f, 0, 10, 4, -3, 10)
        with self.assertRaises(Exception):
            sp_core(self.f, 0, 10, -5, 5, 10)

    def test_result(self):
        out = sp_core(self.f, 1, 10, 4, 5, 10)
        self.assertEqual(out.dim, 4, "Should report 4")
        self.assertEqual(len(out.x_grid), 10, "x grid length must be 10")
        with self.assertRaises(AttributeError):
            print(out.__plot)

    def test_data_fetch(self):
        out = sp_core(self.f, 1, 10, 4, 5, 10)
        expected_fp = [5.5, 5.5, 5.5, 5.5]
        expected_data = [17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5]
        self.assertListEqual(out.data(0)['point'], expected_fp)
        self.assertListEqual(out.data(0)['data'][0], expected_data)
        with self.assertRaises(IndexError):
            print(out.data(0)['data'][11])
        with self.assertRaises(Exception):
            out.data(6)
        with self.assertRaises(Exception):
            out.data(-4)

    def test_json_fetch(self):
        out = sp_core(self.f, 1, 10, 4, 5, 10)
        reload = json.loads(out.to_json())
        self.assertEqual(out.dim, reload['dim'])
        self.assertListEqual(list(out.x_grid), reload['x_grid'])
        self.assertEqual(len(out.data(0)['point']), len(reload['entries'][0]['point']))
        self.assertListEqual(list(out.data(0)['data'][0]), reload['entries'][0]['data'][0])

if __name__ == '__main__':
    unittest.main()
