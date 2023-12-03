import sys
import unittest

import pandas as pd

sys.path.append('./')

from source.data_preparation import (
    USER_ID, ITEM_ID, OLD_NEW_MAP_NAME,
    NEW_OLD_MAP_NAME, N_INDICES,
    get_cont_mapping_struct, 
    update_cont_mapping_struct, 
    get_df_with_updated_indices,
)

class TestDataPreparation(unittest.TestCase):
    def test_get_cont_mapping_struct(self):
        v = [12, 23, 45]
        
        expected = {
            N_INDICES: 3, 
            OLD_NEW_MAP_NAME: {12: 0, 23: 1, 45: 2}, 
            NEW_OLD_MAP_NAME: {0: 12, 1: 23, 2: 45}
        }
        actual = get_cont_mapping_struct(values=v, start=0)
        self.assertDictEqual(actual, expected, msg="Dicts should be equal")

    def test_update_cont_mapping_struct(self):
        v = [12, 23, 45]
    
        expected = {
            N_INDICES: 6, 
            OLD_NEW_MAP_NAME: {12: 0, 23: 1, 45: 2, 99: 3, 101: 4, 505: 5}, 
            NEW_OLD_MAP_NAME: {0: 12, 1: 23, 2: 45, 3: 99, 4: 101, 5: 505}
        }
        actual = get_cont_mapping_struct(values=v, start=0)
        update_cont_mapping_struct(actual, extra_values=[99, 101, 505])
        self.assertDictEqual(actual, expected, msg="Dicts should be equal")

    def test_get_df_with_updated_indices(self):
        df = pd.DataFrame({USER_ID: [12, 23, 45], ITEM_ID: [11, 20, 77]})
        actual_df, actual_mapping = get_df_with_updated_indices(df, (USER_ID, ITEM_ID))
        
        expected_df = pd.DataFrame({USER_ID: [0, 1, 2], ITEM_ID: [0, 1, 2]})
        expected_mapping = {
            USER_ID: {
                N_INDICES: 3,
                OLD_NEW_MAP_NAME: {12: 0, 23: 1, 45: 2},
                NEW_OLD_MAP_NAME: {0: 12, 1: 23, 2: 45},
            },
            ITEM_ID: {
                N_INDICES: 3,
                OLD_NEW_MAP_NAME: {11: 0, 20: 1, 77: 2},
                NEW_OLD_MAP_NAME: {0: 11, 1: 20, 2: 77},
            },
        }
        self.assertDictEqual(actual_mapping, expected_mapping, msg="Dicts should be equal")
        self.assertTrue(((actual_df - expected_df) == 0).all().all())
