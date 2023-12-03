from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from ..data_preparation import (
    USER_ID,
    ITEM_ID,
    RELEVANCE_COLUMN,
    OLD_NEW_MAP_NAME,
    NEW_OLD_MAP_NAME,
    get_df_with_updated_indices
)
from ..random_svd import svd_step
from ..matrix_operations import get_mapped_matrix

class SVD:
    def __init__(
        self,
        rank,
        n_power_iter: int = 1,
        oversampling_factor: int = 2,
        seed: Optional[int] = None,
        dtype=np.float64
    ) -> None:
        self.dtype = dtype
        self.rank = rank
        self.rsvd_params = (n_power_iter, oversampling_factor, seed)

        self.mappings = None
        self.u, self.s, self.vt = None, None, None
        self.history = None # coo - csr matrix of 0/1

    def train(self, data: pd.DataFrame) -> None:
        initial_data, self.mappings = get_df_with_updated_indices(data, (USER_ID, ITEM_ID))
        self.history = (initial_data[USER_ID].values, initial_data[ITEM_ID].values)
        a0 = coo_matrix(
            (
                initial_data[RELEVANCE_COLUMN],
                self.history
            ),
            dtype=self.dtype
        ).tocsr()
        self.u, self.s, self.vt = svd_step(a0, self.rank, seed=self.rsvd_params[2])
        self.user_factors = self.u
        self.item_factors = (self.s[:, np.newaxis] * self.vt)

    def get_factors(self):
        return self.u, self.s, self.vt

    def get_u(self):
        return self.u
    
    def get_v(self):
        return self.vt.T
    
    def get_n_users(self) -> int:
        return self.u.shape[0]

    def get_n_items(self) -> int:
        return self.vt.shape[1]

    def _raw_history_to_csr(self) -> csr_matrix:
        rows, cols = self.history
        return get_mapped_matrix(
            np.ones_like(rows),
            rows, 
            cols, 
            shape=(self.u.shape[0], self.vt.shape[1]), 
            mtx_format='csr'
        )

    def _recommend_user(self, user_ind: int) -> np.ndarray:
        return self.item_factors.T.dot(self.user_factors[user_ind])

    def recommend(
        self, 
        users: List, 
        k: int, 
        filter_viewed: bool = True, 
        internal: bool = False
    ):
        _, n_items = self.vt.shape  
        items_array = None if not filter_viewed else np.arange(n_items)  
        mapped_users = [self.mappings[USER_ID][OLD_NEW_MAP_NAME][uid] for uid in users]
        k = min(k, n_items) 
        preferences = None if not filter_viewed else self._raw_history_to_csr()
        
        results = np.empty((len(mapped_users), k))  
        for i, user_ind in enumerate(mapped_users):  
            rec = self._recommend_user(user_ind) 
            if filter_viewed:
                u_viewed_mask = preferences[user_ind].A.squeeze().astype(bool)
                candidate_items = np.argpartition(rec[~u_viewed_mask], -k)[-k:][::-1]
                candidate_items = items_array[~u_viewed_mask][candidate_items]
            else:
                candidate_items = np.argpartition(rec, -k)[-k:][::-1]
            if internal:
                results[i] = candidate_items
            else:
                results[i] = np.array([self.mappings[ITEM_ID][NEW_OLD_MAP_NAME][j] for j in candidate_items])  
        return results.astype(int)
