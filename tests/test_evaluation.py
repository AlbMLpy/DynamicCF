import sys
import unittest

import numpy as np

sys.path.append('./')

from source.evaluation import hr, mrr, wji_sim

class TestEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        self.rec_array = np.array(
            [
                [1, 2, 3], 
                [4, 5, 6], 
                [7, 8, 9],
            ]
        )
        self.test_items = np.array([3, 7, 8])

    def test_hr(self):
        expected = 2/3
        actual = hr(self.rec_array, self.test_items)
        self.assertAlmostEqual(actual, expected, msg=f"Values should be almost equal: {expected = }, {actual = }")

    def test_mrr(self):
        expected = 0.27777777777777773
        actual = mrr(self.rec_array, self.test_items)
        self.assertAlmostEqual(actual, expected, msg=f"Values should be almost equal: {expected = }, {actual = }")

    def test_wji_sim(self):
        # prepare the data:
        rec_array_u = np.array(
            [
                [2, 3, 4],
                [0, 2, 3], 
                [4, 3, 2]
            ]
        )
        rec_array_v = np.array(
            [
                [2, 3, 4],
                [2, 3, 1], 
                [4, 2, 3]
            ]
        )
        expected = 0.7091503267973857
        actual = wji_sim(rec_array_u, rec_array_v, N=5)
        self.assertAlmostEqual(actual, expected, msg=f"Values should be almost equal: {expected = }, {actual = }")
