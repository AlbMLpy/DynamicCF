import sys
import unittest
from collections import deque

import numpy as np

sys.path.append('./')

from source.models.tdrec import (
    get_global_attention, 
    _get_pos_items_arrays, 
    TDRec
)
from source.ti_data_processing import update_user_seq_history

class TestTDRec(unittest.TestCase):

    def setUp(self) -> None:
        self.max_len_history = 4
        self.model = TDRec(
            rank=(2, 2, 2),
            seq_len=self.max_len_history,
            att_f=2,
            seed=13
        )
        self.model.w = np.array(
            [
                [1, 2], 
                [2, 3], 
                [3, 4], 
                [1, 1],
            ]
        )
        self.model.v = np.array(
            [
                [1, 1], 
                [2, 2], 
                [3, 3], 
                [4, 4],
                [5, 5]
            ]
        )
        self.model.seq_history = {}
        update_user_seq_history(
            np.array(
                [
                    [0, 3, 0],
                    [0, 2, 1],
                    [0, 1, 2]
                ]
            ),
            self.model.seq_history,
            self.max_len_history,
        )
        self.model.aw = self.model.attention_mtx.dot(self.model.w)
        self.model.last_pos_emb = self.model.attention_mtx_inv.T.dot(self.model.w)[-1]

    def test_get_global_attention(self):
        actual = get_global_attention(4, 1)
        expected = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.5, 1.0, 0.0, 0.0],
                [0.33333333, 0.5, 1.0, 0.0],
                [0.25, 0.33333333, 0.5, 1.0]
            ]
        )
        self.assertLess((actual - expected).sum(), 1e-8)

    def test_get_pos_items_arrays(self):
        actual_pos, actual_items = _get_pos_items_arrays([4, 5, 10, 11, 50], 4)
        expected_pos, expected_items = (np.array([2, 1, 0]), np.array([50, 11, 10]))
        self.assertTrue((actual_pos == expected_pos).all())
        self.assertTrue((actual_items == expected_items).all())

    def test_recommend_user(self):
        actual = self.model._recommend_user(0)
        expected = np.array(
            [
                92.83333333, 185.66666667, 
                278.5, 371.33333333, 464.16666667,
            ]
        )
        self.assertAlmostEqual(np.linalg.norm(actual - expected), 0)
