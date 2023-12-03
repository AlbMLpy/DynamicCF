from typing import Optional, Callable

import numpy as np
import pandas as pd
from scipy.sparse import diags, csr_matrix
from scipy.linalg import inv

from ..data_preparation import (
    USER_ID,
    ITEM_ID,
    POSITION_ID,
    RELEVANCE_COLUMN,
    OLD_NEW_MAP_NAME,
    NEW_OLD_MAP_NAME,
    get_df_with_updated_indices
)
from ..rp_hooi import ga_satf, TuckerFactorsCore
from ..matrix_operations import get_mapped_matrix
from ..ti_data_processing import update_user_seq_history, PAD_VALUE

def get_global_attention(size: int, power: float) -> np.ndarray:
    """
    Get a lower triangular attention matrix of definite size such that:
    a_k = k^{-power}, k = 1, ..., size, where k = 1 - main diagonal, 
    k = 2 - the next diagonal to the left and so on.

    Parameters
    ----------
    size : int
        Attention square matrix size.
    power : float
        Factor power = f ≥ 0. When f = 0, all preceding positions are
        equally important for an observation at the current position,
        and with f > 0 items that are more distant in the sequence
        from the current one get lower attention.

    Returns
    -------
    output : np.ndarray
        Lower triangular attention matrix

    """
    diag_list = [[1 / i**power]*(size - i + 1) for i in range(1, size + 1)]
    offsets = [i for i in range(0, -size, -1)]
    return diags(diag_list, offsets).A

def _get_pos_items_arrays(
    seq: list[int], 
    max_seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get positions and items out of user history sequence shifted by 1.
    For example: 
        - Input: seq = ['a', 'c', 'd', 'm', 'p'], max_seq_len = 4
        - Output: (np.array([2, 1, 0]), np.array(['p', 'm', 'd']))
        - Note: without shifting the results would be: 
            (np.array([3, 2, 1, 0]), np.array(['p', 'm', 'd', 'c']))
    """
    positions = []
    items = []
    for i in range(len(seq)):
        pos = max_seq_len - i - 2 
        if pos < 0:
            break
        positions.append(pos)
        items.append(seq[-i-1])
    return np.array(positions), np.array(items)

class TDRec:
    """ 
    Sequential Recommender System based on Tucker Decomposition of 
    User-Item-Position data tensor with attention.

    Parameters
    ----------
    rank : tuple[int, int, int]
        TODO
    seq_len : int
        TODO
    att_f : float, optional, default: None
        TODO
    n_iter : int, default: 25
        TODO
    growth_tol : float, default: 0.01
        TODO
    n_power_iter : int, default: 1
        TODO
    oversampling_factor : int, default: 2
        TODO
    seed : int, optional, default: None
        TODO
    parallel : bool, default: False
        TODO
    dtype: float, default: np.float64
        TODO
    force_n_iter : bool
        Use strictly n_iter cycles in a loop of a model.

    Attributes
    ----------
    u : np.ndarray 
        Users embeddings (internal indexation)
    v : np.ndarray
        Items embeddings (internal indexation)
    w : np.ndarray
        Positions embeddings (internal indexation)
    mappings: dict[str, dict[str, dict[int, int]]]
        Mappings between users/items external and internal indices.

    Notes
    -----
    TDRec - Tucker Decomposition Recommender

    References
    ----------
    1. https://arxiv.org/abs/2212.05720 - Look for (GA-SATF)


    Examples
    --------
    TODO
    """
    def __init__(
        self, 
        rank: tuple[int, int, int],
        seq_len: int,
        att_f: Optional[float] = None,
        n_iter: int = 25, 
        growth_tol: float = 0.01,
        n_power_iter: int = 1,
        oversampling_factor: int = 2,
        seed: Optional[int] = None,
        parallel: bool = False,
        dtype=np.float64,
        force_n_iter: bool = False,
    ) -> None:
        self.rank = rank
        self.seq_len = seq_len
        self.n_iter = n_iter
        self.growth_tol = growth_tol
        self.n_power_iter = n_power_iter
        self.oversampling_factor = oversampling_factor
        self.seed = seed
        self.parallel = parallel
        self.dtype = dtype
        self.force_n_iter = force_n_iter
        self.att_f = att_f
        self.attention_mtx = (
            np.eye(seq_len) 
            if att_f is None else get_global_attention(seq_len, att_f)
        )
        self.attention_mtx_inv = (
            np.eye(seq_len) 
            if att_f is None else inv(self.attention_mtx)
        )
        # User/Item internal/external ids transform:
        self.mappings = None
        # Users, Items, Positions factors and Tucker core
        self.u, self.v, self.w, self.core = None, None, None, None
        # Attention matrix by position matrix:
        self.aw = None 
        self.last_pos_emb = None
        self.history = None # coo - csr matrix of 0/1
        self.seq_history = {} # user-item sequential history of last self.seq_len interactions

    def train(
        self, 
        data: pd.DataFrame,
        exit_callback: Optional[Callable[[TuckerFactorsCore], bool]] = None,
    ) -> None:
        """
        Train TDRec model parameters: user, item, position embeddings using 'data'.
        
        Parameters
        ----------
        data : pandas.Dataframe
            TODO
        exit_callback : callable, optional, default: None
            Return True if factors and core are "good", else False
        
        Returns
        -------
            None
        """
        # Map user/item ids into external indexation, 
        # return new dataframe and mappings:
        initial_data, self.mappings = get_df_with_updated_indices(
            data, (USER_ID, ITEM_ID)
        )
        # Prepare ndarray representation of data tensor for computations:
        inds = (
            initial_data[[USER_ID, ITEM_ID, POSITION_ID]]
            .sort_values(by=[USER_ID, POSITION_ID])
            .to_numpy()
        )
        vals = initial_data[RELEVANCE_COLUMN].to_numpy()
        shape = tuple(inds.max(axis=0) + 1)
        # Prepare history of interactions for recs and further calculations:
        self.history = (initial_data[USER_ID].values, initial_data[ITEM_ID].values)
        update_user_seq_history(
            inds, 
            self.seq_history, 
            self.seq_len
        )
        # Get user, item, position embeddings:
        self.u, self.v, self.w, self.core = ga_satf(
            inds, vals, shape, self.rank, 
            attention_mtx=self.attention_mtx,
            n_iter=self.n_iter,
            growth_tol=self.growth_tol, 
            n_power_iter=self.n_power_iter,
            oversampling_factor=self.oversampling_factor,
            seed=self.seed,
            parallel=self.parallel,
            exit_callback=exit_callback,
            force_n_iter=self.force_n_iter,
        )
        self.aw = self.attention_mtx.dot(self.w)
        self.last_pos_emb = self.attention_mtx_inv.T.dot(self.w)[-1]

    def get_factors(self):
        return self.u, self.v, self.w

    def get_u(self):
        return self.u

    def get_v(self):
        return self.v

    def get_n_users(self) -> int:
        return self.u.shape[0]

    def get_n_items(self) -> int:
        return self.v.shape[0]

    def _raw_history_to_csr(self) -> csr_matrix:
        """
        Transform user-item interactions history into binary matrix
        in CSR format.

        NOTE: UPDATE THIS LOGIC! MAY BREAK DOWN IF TOO MANY INTERACTIONS!!!
        """
        rows, cols = self.history
        return get_mapped_matrix(
            np.ones_like(rows),
            rows, 
            cols, 
            shape=(self.u.shape[0], self.v.shape[0]), 
            mtx_format='csr'
        )

    def _recommend_user(self, user_ind: int) -> np.ndarray:
        """
        Get relevance scores for all the items for a particular user 
        using the following formula: toprec_{GA-SATF}(P, n) = arg_max VV^\topPSAWŵ_{k},
        where V = self.v, PS - constructed using self.seq_history, AW = self.aw,
        ŵ_{k} = self.w[-1] (last position).
        """
        n_pos, _ = self.w.shape
        user_seq = [v for v in self.seq_history[user_ind] if v != PAD_VALUE]
        positions, items = _get_pos_items_arrays(user_seq, n_pos)
        _w = self.aw.dot(self.last_pos_emb)
        res = self.v.dot(self.v[items].T.dot(_w[positions]))
        return res

    def recommend(
        self, 
        users: list, 
        k: int, 
        filter_viewed: bool = True,
        internal: bool = False
    ) -> np.ndarray:
        """
        Generate 'k' recommendations for particular 'users'.

        Parameters
        ----------
        users : list[int]
            TODO
        k : int
            TODO
        filter_viewed: bool, default: True
            TODO
        
        Returns
        -------
        output : np.ndarray
            TODO
        """
        n_items, _ = self.v.shape  
        k = min(k, n_items) 
        items_array = None if not filter_viewed else np.arange(n_items)  
        preferences = None if not filter_viewed else self._raw_history_to_csr()
        mapped_users = [self.mappings[USER_ID][OLD_NEW_MAP_NAME][uid] for uid in users]
        # Get recommendations per user:
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
