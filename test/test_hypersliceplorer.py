import unittest
import json
import math

from pysliceplorer import hypersliceplorer_core as hsp_core


class TestHypersliceplorer(unittest.TestCase):
    vertices = [
        [-1, 1, 1],
        [1, 1, 1],
        [-1, -1, 1],
        [1, -1, 1],
        [-1, 1, -1],
        [1, 1, -1],
        [-1, -1, -1],
        [1, -1, -1]
    ]

    config = [
        [1, 0, 2],  # top face
        [3, 1, 2],
        [6, 4, 7],  # bottom face
        [7, 5, 4],
        [6, 2, 4],  # left face
        [2, 4, 0],
        [7, 3, 2],  # front face
        [6, 7, 2],
        [7, 5, 1],  # right face
        [7, 1, 3],
        [5, 4, 1],  # back face
        [1, 4, 0]
    ]

    def test_min_max(self):
        with self.assertRaises(Exception):
            hsp_core(self.vertices, self.config, 20, 10, 5)

    def test_nfpoint_consistency(self):
        with self.assertRaises(Exception):
            hsp_core(self.vertices, self.config, -1, 1, -5)

    def test_result(self):
        def is_between(p1, p2, s):
            def distance(a, b):
                return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

            epsilon = 1e-6
            check = distance(p1, p2) - (distance(p1, s) + distance(p2, s))
            return -epsilon < check < epsilon

        out = hsp_core(self.vertices, self.config, -1, 1, 5)
        self.assertTrue(out.data_by_axes(0, 1))
        self.assertEqual(len(out.data_by_axes(0, 1)), 5)
        self.assertEqual(len(out.data_by_axes(0, 1)[(0, 0, 0)]), 8)
        segments = []
        for seg in out.data_by_axes(0, 1)[(0, 0, 0)]:
            segments.append([(seg['p1_1'], seg['p1_2']), (seg['p2_1'], seg['p2_2'])])

        test_point = (0.5, 1)
        flag = False
        for seg in segments:
            if is_between(seg[0], seg[1], test_point):
                flag = True
        self.assertTrue(flag)

        test_point = (0.5, 1.5)
        flag = False
        for seg in segments:
            if is_between(seg[0], seg[1], test_point):
                flag = True
        self.assertFalse(flag)

    def test_data_fetch(self):
        out = hsp_core(self.vertices, self.config, -1, 1, 5)
        self.assertListEqual(out.vertices, self.vertices)
        self.assertListEqual(out.config, self.config)
        with self.assertRaises(IndexError):
            out.data_by_axes(0, 4)
        with self.assertRaises(IndexError):
            out.data_by_axes(0, -2)

    def test_json_fetch(self):
        out = hsp_core(self.vertices, self.config, -1, 1, 5)
        reload = json.loads(out.to_json())
        self.assertEqual(reload['size'], 5)
        self.assertListEqual(reload['vertices'], self.vertices)
        self.assertListEqual(reload['config'], self.config)

if __name__ == '__main__':
    unittest.main()
