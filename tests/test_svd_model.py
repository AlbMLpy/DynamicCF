import sys
import unittest

import numpy as np
from scipy.sparse import csr_matrix

sys.path.append('./')

from source.models.svd import SVD
from source.data_preparation import (
    USER_ID,
    ITEM_ID,
    OLD_NEW_MAP_NAME,
    NEW_OLD_MAP_NAME,
)

class TestSVD(unittest.TestCase):
    def setUp(self) -> None:
        self.u = np.array(
            [
                [ 0.37427841, -0.30118744, -0.96556633],
                [ 2.29874821, -0.21836592, -0.44987196],
                [-0.93689866, -0.35001607, -1.20881597],
                [-0.84424139, 0.45886554, 0.99219709],
                [-0.23778262, -0.20085885, -1.31348285]
            ]
        )  
        self.vt = np.array(
            [
                [-0.8064354, 1.06088316, 0.30051897, 0.052212, 0.26933904, -0.85300425],
                [-0.17560066, 0.70145484, 1.35287405, 0.28870878, -1.090444, -0.55996878],
                [0.11033713, -1.45870458, 0.64202769, -0.97937828, 0.22735107, -0.84464499]
            ]
        )  
        self.m, self.n, self.rank = self.u.shape[0], self.vt.shape[1], self.vt.shape[0]
  
        self.mappings = {  
            USER_ID: {  
                OLD_NEW_MAP_NAME: {i + 100: i for i in range(self.m)},  
                NEW_OLD_MAP_NAME: {i: i + 100 for i in range(self.m)}  
            },  
            ITEM_ID: {  
                OLD_NEW_MAP_NAME: {i + 200: i for i in range(self.n)},  
                NEW_OLD_MAP_NAME: {i: i + 200 for i in range(self.n)}  
            }  
        }  
 
        self.k = 2
        self.users_to_recommend = [100, 101, 102, 103]  

        rows = np.array([0, 0, 1, 1, 2, 2, 2, 2, 3, 3])
        cols = np.array([1, 3, 1, 2, 1, 2, 3, 5, 1, 5])
        self.history = (rows, cols)

    def test_recommend_with_filter(self):
        svd_model = SVD(self.rank, seed=13)
        svd_model.u = self.u
        svd_model.vt = self.vt
        svd_model.mappings = self.mappings
        svd_model.history = self.history
        svd_model.user_factors = self.u
        svd_model.item_factors = self.vt

        actual = svd_model.recommend(
            self.users_to_recommend, 
            self.k,  
            filter_viewed=True,
        )  
        expected = np.array(
            [
                [205, 204],
                [204, 203],
                [200, 204],
                [202, 200],
            ]
        )
        self.assertEqual((actual - expected).sum(), 0)

    def test_recommend_no_filter(self):
        svd_model = SVD(self.rank, seed=13)
        svd_model.u = self.u
        svd_model.vt = self.vt
        svd_model.mappings = self.mappings
        svd_model.history = self.history
        svd_model.user_factors = self.u
        svd_model.item_factors = self.vt

        actual = svd_model.recommend(
            self.users_to_recommend, 
            self.k,  
            filter_viewed=False,
        ) 
        expected = np.array(
            [
                [201, 203],
                [201, 204],
                [205, 203],
                [202, 200],
            ]
        )
        self.assertEqual((actual - expected).sum(), 0)
