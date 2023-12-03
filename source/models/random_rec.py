import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ..data_preparation import (
    USER_ID,
    ITEM_ID,
    POSITION_ID,
    OLD_NEW_MAP_NAME,
    NEW_OLD_MAP_NAME,
    get_df_with_updated_indices
)

from ..matrix_operations import get_mapped_matrix

class RandomRec:
    def __init__(self, seed: int = 0) -> None:
        self.mappings = None
        self.n_users, self.n_items = None, None
        self.history = None # coo - csr matrix of 0/1
        self.rs = np.random.RandomState(seed)

    def train(self, data: pd.DataFrame) -> None:
        initial_data, self.mappings = get_df_with_updated_indices(
            data, (USER_ID, ITEM_ID)
        )
        self.history = (initial_data[USER_ID].values, initial_data[ITEM_ID].values)
        self.n_users, self.n_items, _ = tuple(initial_data[[USER_ID, ITEM_ID, POSITION_ID]].max(axis=0) + 1)
    
    def _raw_history_to_csr(self) -> csr_matrix:
        rows, cols = self.history
        return get_mapped_matrix(
            np.ones_like(rows),
            rows, 
            cols, 
            shape=(self.n_users, self.n_items), 
            mtx_format='csr'
        )
    
    def get_n_users(self) -> int:
        return self.n_users

    def get_n_items(self) -> int:
        return self.n_items

    def _recommend_user(self) -> np.ndarray:
        return self.rs.randn(self.n_items)

    def recommend(
        self, 
        users: list, 
        k: int, 
        filter_viewed: bool = True, 
        internal: bool = False
    ): 
        items_array = None if not filter_viewed else np.arange(self.n_items)  
        mapped_users = [self.mappings[USER_ID][OLD_NEW_MAP_NAME][uid] for uid in users]
        k = min(k, self.n_items) 
        preferences = None if not filter_viewed else self._raw_history_to_csr()
        
        results = np.empty((len(mapped_users), k))  
        for i, user_ind in enumerate(mapped_users): 
            rec = self._recommend_user()
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
