from typing import Optional, Any

import numpy as np
import pandas as pd

from .tdrec import TDRec
from ..data_preparation import (
    USER_ID,
    ITEM_ID,
    POSITION_ID,
    RELEVANCE_COLUMN,
    OLD_NEW_MAP_NAME,
    get_df_with_updated_indices,
    update_cont_mapping_struct,
    map_df_columns,
)
from ..rp_hooi import ga_satf, TuckerFactors
from ..ti_data_processing import update_user_seq_history

class TDRecRe(TDRec):
    """ 
    Sequential Recommender System based on Tucker Decomposition of 
    User-Item-Position data tensor with attention with reinitialization.

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
        force_n_iter : bool = False,
    ) -> None:
        super().__init__(
            rank,
            seq_len,
            att_f,
            n_iter,
            growth_tol,
            n_power_iter,
            oversampling_factor,
            seed,
            parallel,
            dtype,
            force_n_iter,
        )

    def train(
        self, 
        data: pd.DataFrame, 
        factors_init_list: Optional[list[TuckerFactors]] = None, 
        mappings: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Train TDRec model parameters: user, item, position embeddings using 'data'.
        
        Parameters
        ----------
        data : pandas.Dataframe
            TODO
        
        Returns
        -------
            None
        """
        if (factors_init_list is not None) and (mappings is not None):
            # Get new users ids, update mappings:
            new_users = [
                ent for ent in data[USER_ID].unique() 
                if mappings[USER_ID][OLD_NEW_MAP_NAME].get(ent) is None
            ]
            update_cont_mapping_struct(mappings[USER_ID], new_users)
            # Get new items ids, update mappings:
            new_items = [
                ent for ent in data[ITEM_ID].unique() 
                if mappings[ITEM_ID][OLD_NEW_MAP_NAME].get(ent) is None
            ]
            update_cont_mapping_struct(mappings[ITEM_ID], new_items)
            # Map user/item ids into internal repr:
            initial_data = map_df_columns(data, (USER_ID, ITEM_ID), mappings)
            self.mappings = mappings
        else:   
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
            init_factors_list=factors_init_list,
            force_n_iter=self.force_n_iter,
        )
        self.aw = self.attention_mtx.dot(self.w)
        self.last_pos_emb = self.attention_mtx_inv.T.dot(self.w)[-1]
