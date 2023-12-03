import sys
import unittest
from copy import deepcopy
from collections import deque

import numpy as np
import pandas as pd

sys.path.append('./')

from source.ti_data_processing import (
    update_user_seq_history, 
    calculate_delta,
    add_order_cropped,
    get_df_with_cropped_pos_column
)
from source.data_preparation import (
    USER_ID, ITEM_ID
)
POSITION_ID = 'position'

class TestTI_Data_Processing(unittest.TestCase):
    def setUp(self):
        self.max_len_history = 4

        self.a0 = pd.DataFrame(
            {
                USER_ID: [0, 0, 1, 1, 2, 2, 3, 3],
                ITEM_ID: [1, 0, 0, 2, 2, 1, 1, 2],
                POSITION_ID: [0, 1, 0, 1, 0, 1, 0, 1],
            }
        )
        self.a1 = pd.DataFrame(
            {
                USER_ID: [0, 0, 1, 2, 2],
                ITEM_ID: [0, 1, 1, 2, 1],
                POSITION_ID: [2, 3, 2, 2, 3],
            }
        )
        self.a2 = pd.DataFrame(
            {
                USER_ID: [0, 1, 1, 2],
                ITEM_ID: [2, 0, 2, 2],
                POSITION_ID: [4, 3, 4, 4],
            }
        )
        self.a3 = pd.DataFrame(
            {
                USER_ID: [0, 0, 1, 1, 2, 3],
                ITEM_ID: [0, 1, 1, 0, 0, 0],
                POSITION_ID: [5, 6, 5, 6, 5, 2],
            }
        )
        self.a_list = [self.a0, self.a1, self.a2, self.a3]

    def test_update_user_seq_history(self):
        user_seq_history = {}
        for ai in self.a_list[:-1]:
            ai = ai.sort_values(by=[USER_ID, POSITION_ID])
            update_user_seq_history(ai.values, user_seq_history, self.max_len_history)
        
        # update 2 times:
        expected = {
            0: deque([0, 0, 1, 2], maxlen=self.max_len_history),
            1: deque([2, 1, 0, 2], maxlen=self.max_len_history),
            2: deque([1, 2, 1, 2], maxlen=self.max_len_history),
            3: deque([-1, -1, 1, 2], maxlen=self.max_len_history),
        }
        actual = user_seq_history
        self.assertDictEqual(actual, expected, msg=f"dict: {actual} != dict: {expected}")

        # update one more time:
        a3 = self.a3.sort_values(by=[USER_ID, POSITION_ID])
        update_user_seq_history(a3.values, user_seq_history, self.max_len_history)
        expected = {
            0: deque([1, 2, 0, 1], maxlen=self.max_len_history),
            1: deque([0, 2, 1, 0], maxlen=self.max_len_history),
            2: deque([2, 1, 2, 0], maxlen=self.max_len_history),
            3: deque([-1, 1, 2, 0], maxlen=self.max_len_history),
        }
        actual = user_seq_history
        self.assertDictEqual(actual, expected, msg=f"dict: {actual} != dict: {expected}")

    def test_calculate_delta_1(self):
        user_seq_history = {}
        a0 = self.a0.sort_values(by=[USER_ID, POSITION_ID])
        update_user_seq_history(a0.values, user_seq_history, self.max_len_history)

        user_seq_history_old = {u: list(v) for u, v in user_seq_history.items()}#deepcopy(user_seq_history)

        a1 = self.a1.sort_values(by=[USER_ID, POSITION_ID])
        update_user_seq_history(a1.values, user_seq_history, self.max_len_history)

        inds, vals = calculate_delta(user_seq_history, user_seq_history_old)
        
        expected = np.sort(
            np.array(
                [
                    [0, 1, 0, 1],
                    [0, 0, 1, 1],
                    [0, 0, 2, 1],
                    [0, 1, 2, -1],
                    [0, 0, 3, -1],
                    [0, 1, 3, 1], 
                    [1, 0, 1, 1],
                    [1, 0, 2, -1],
                    [1, 2, 2, 1],
                    [1, 1, 3, 1],
                    [1, 2, 3, -1],
                    [2, 2, 0, 1],
                    [2, 1, 1, 1],
                ]
            ),
            axis=0,
        )
        actual = np.sort(np.concatenate([inds, vals[:, np.newaxis]], axis=1), axis=0)
        self.assertTrue((actual == expected).all(), msg=f"actual: {actual} != expected: {expected}")

    def test_calculate_delta_2(self):
        user_seq_history = {}
        for ai in self.a_list[:2]:
            ai = ai.sort_values(by=[USER_ID, POSITION_ID])
            update_user_seq_history(ai.values, user_seq_history, self.max_len_history)

        user_seq_history_old = {u: list(v) for u, v in user_seq_history.items()}#deepcopy(user_seq_history)

        a = self.a_list[2].sort_values(by=[USER_ID, POSITION_ID])
        update_user_seq_history(a.values, user_seq_history, self.max_len_history)

        inds, vals = calculate_delta(user_seq_history, user_seq_history_old)
        
        expected = np.sort(
            np.array(
                [
                    [0, 0, 0, 1],
                    [0, 1, 0, -1],
                    [0, 0, 2, -1],
                    [0, 1, 2, 1],
                    [0, 1, 3, -1],
                    [0, 2, 3, 1], 
                    [1, 2, 0, 1],
                    [1, 0, 1, -1],
                    [1, 1, 1, 1],
                    [1, 0, 2, 1],
                    [1, 2, 2, -1],
                    [1, 1, 3, -1],
                    [1, 2, 3, 1],
                    [2, 1, 0, 1],
                    [2, 2, 0, -1],
                    [2, 1, 1, -1],
                    [2, 2, 1, 1],
                    [2, 1, 2, 1],
                    [2, 2, 2, -1],
                    [2, 1, 3, -1],
                    [2, 2, 3, 1],
                ]
            ),
            axis=0,
        )
        actual = np.sort(np.concatenate([inds, vals[:, np.newaxis]], axis=1), axis=0)
        self.assertTrue((actual == expected).all(), msg=f"actual: {actual} != expected: {expected}")

    def test_calculate_delta_3(self):
        user_seq_history = {}
        for ai in self.a_list[:3]:
            ai = ai.sort_values(by=[USER_ID, POSITION_ID])
            update_user_seq_history(ai.values, user_seq_history, self.max_len_history)

        user_seq_history_old = deepcopy(user_seq_history)

        a = self.a_list[3].sort_values(by=[USER_ID, POSITION_ID])
        update_user_seq_history(a.values, user_seq_history, self.max_len_history)

        inds, vals = calculate_delta(user_seq_history, user_seq_history_old)
        
        expected = np.sort(
            np.array(
                [
                    [0, 0, 0, -1],
                    [0, 1, 0, 1],
                    [0, 0, 1, -1],
                    [0, 2, 1, 1],
                    [0, 0, 2, 1],
                    [0, 1, 2, -1], 
                    [0, 1, 3, 1],
                    [0, 2, 3, -1], 
                    [1, 0, 0, 1],
                    [1, 2, 0, -1],
                    [1, 1, 1, -1],
                    [1, 2, 1, 1],
                    [1, 0, 2, -1],
                    [1, 1, 2, 1],
                    [1, 0, 3, 1],
                    [1, 2, 3, -1],
                    [2, 1, 0, -1],
                    [2, 2, 0, 1],
                    [2, 1, 1, 1],
                    [2, 2, 1, -1],
                    [2, 1, 2, -1],
                    [2, 2, 2, 1],
                    [2, 0, 3, 1],
                    [2, 2, 3, -1],
                    [3, 1, 1, 1],
                    [3, 1, 2, -1],
                    [3, 2, 2, 1],
                    [3, 0, 3, 1],
                    [3, 2, 3, -1],
                ]
            ),
            axis=0,
        )
        actual = np.sort(np.concatenate([inds, vals[:, np.newaxis]], axis=1), axis=0)
        self.assertTrue((actual == expected).all(), msg=f"actual: {actual} != expected: {expected}")

    def test_add_order_cropped(self):
        data_dict = {
            USER_ID: [0, 0, 0, 0, 0, 0],
            ITEM_ID: [1, 2, 3, 4, 3, 2]
        }
        input_df = pd.DataFrame(data_dict)

        expected = pd.DataFrame(
            {
                **data_dict,
                **{POSITION_ID: [-1, -1, 0, 1, 2, 3]}
            }
        )
        actual = add_order_cropped(input_df, max_len=4)
        self.assertTrue((actual == expected).all().all(), msg=f"actual: {actual} != expected: {expected}")

    def test_get_df_with_cropped_pos_column(self):
        data_dict = {
            USER_ID: [0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3],
            ITEM_ID: ['a', 'b', 'c', 'd', 'A', 'B', 'C', 'D', 'a1', 'a2', 'a3'],
        }
        input_df = pd.DataFrame(data_dict)

        expected = pd.DataFrame(
            {
                USER_ID: [0, 0, 0, 1, 1, 1, 3, 3, 3],
                ITEM_ID: ['b', 'c', 'd', 'B', 'C', 'D', 'a1', 'a2', 'a3'],
                POSITION_ID: [0, 1, 2, 0, 1, 2, 0, 1, 2],
            }
        )
        actual = get_df_with_cropped_pos_column(input_df, max_seq_len=3, sort_bool=False)
        self.assertTrue((actual.values == expected.values).all(), msg=f"actual: {actual} != expected: {expected}")
