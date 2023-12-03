import sys
import unittest

import numpy as np
import pandas as pd

sys.path.append('./')

from source.models.psi import _get_divide_df_mask
from source.data_preparation import (
    USER_ID,
    ITEM_ID,
    OLD_NEW_MAP_NAME
)

class TestPSI(unittest.TestCase):
    def setUp(self) -> None:
        self.df_div_mask = pd.DataFrame(
            {
                USER_ID: np.arange(10), 
                ITEM_ID: np.arange(10) + 1,
            }
        )
        self.mappings_div_mask = {
            USER_ID: {
                OLD_NEW_MAP_NAME: {i: 100*i for i in np.arange(1, 5)},
            },
            ITEM_ID: {
                OLD_NEW_MAP_NAME: {i: 200*i for i in np.arange(1, 5)},
            }
        }

    def test_get_divide_df_mask_new_users(self):
        expected = np.array([True] + [False]*9)
        actual = _get_divide_df_mask(self.df_div_mask, self.mappings_div_mask, by="new_users")
        self.assertTrue((actual == expected).all())

    def test_get_divide_df_mask_new_items(self):
        expected = np.array([False]*4 + [True] + [False]*5)
        actual = _get_divide_df_mask(self.df_div_mask, self.mappings_div_mask, by="new_items")
        self.assertTrue((actual == expected).all())

    def test_get_divide_df_mask_old(self):
        expected = np.array([False] + [True]*3 + [False]*6)
        actual = _get_divide_df_mask(self.df_div_mask, self.mappings_div_mask, by="old")
        self.assertTrue((actual == expected).all())

    def test_get_divide_df_mask_new(self):
        expected = np.array([False]*5 + [True]*5)
        actual = _get_divide_df_mask(self.df_div_mask, self.mappings_div_mask, by="new")
        self.assertTrue((actual == expected).all())
