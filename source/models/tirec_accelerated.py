from typing import Optional

import numpy as np
import pandas as pd

from .tdrec import TDRec
from ..tucker_integrator import (
    tucker_intergrator,
    update_tucker_new_matrix,
)
from ..tensor_operations import (
    tensordot,
    unfold_dense,
    fold_dense,
    unfold_sparse,
)
from ..ti_data_processing import (
    update_user_seq_history, 
    calculate_delta,
    get_df_with_cropped_pos_column,
)
from ..data_preparation import (
    USER_ID, ITEM_ID, POSITION_ID,
    OLD_NEW_MAP_NAME, 
    update_cont_mapping_struct, 
    map_df_columns,
)

class TIRecA(TDRec):
    """ 
    Sequential Recommender System based on Tucker Integrator  of 
    User-Item-Position data tensor with attention. Accelerated version using 
    zero-vectors embeddings for new entities.

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
    att_light_mode : ???
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
    2. 

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
        att_light_mode: bool = False,
        force_n_iter: bool = False
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
        self.att_light_mode = att_light_mode

        # Several modes for initialization and processing of new users/items:
        # - 0 -> use zero embeddings as initialization (update is done by PSI)
        # - float x > 0 -> use random initialization (update is done by PSI)
        self._init_new_embeddings = 0 
        self._nu_ni_process_integrator = False

    def _update_history(self, data: pd.DataFrame):# UPDATE THERE MIGHT BE PROBELMS WITH DUPLICATES!!!!!!!!!!!!
        modes = (USER_ID, ITEM_ID)
        temp_history = []
        for i in range(len(self.history)):
            temp_history.append(
                np.concatenate(
                (
                    self.history[i], 
                    data[modes[i]]
                )
            )
        )
        self.history = (temp_history[0], temp_history[1])

    def preprocess_raw_input_data(
        self, 
        data: pd.DataFrame, 
        sort_bool: bool = True
    ) -> tuple[pd.DataFrame, list[int], list[int]]:
        """
        Leave only the latest user-item interactions and map ids into internal indicies.
        """
        # Add new 'position' column and leave the latest interactions:
        chunk = get_df_with_cropped_pos_column(data, self.seq_len, sort_bool)
        # Get new users ids, update mappings:
        new_users = [
            ent for ent in chunk[USER_ID].unique() 
            if self.mappings[USER_ID][OLD_NEW_MAP_NAME].get(ent) is None
        ]
        update_cont_mapping_struct(self.mappings[USER_ID], new_users)
        # Get new items ids, update mappings:
        new_items = [
            ent for ent in chunk[ITEM_ID].unique() 
            if self.mappings[ITEM_ID][OLD_NEW_MAP_NAME].get(ent) is None
        ]
        update_cont_mapping_struct(self.mappings[ITEM_ID], new_items)
        # Map new user to internal repr:
        new_users = [self.mappings[USER_ID][OLD_NEW_MAP_NAME][ent] for ent in new_users]
        new_items = [self.mappings[ITEM_ID][OLD_NEW_MAP_NAME][ent] for ent in new_items]
        # Map user/item ids into internal repr:
        chunk = map_df_columns(chunk, (USER_ID, ITEM_ID), self.mappings)
        return chunk, new_users, new_items
    
    def get_train_tensor(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Note: This function has a side effect: update self.seq_history!!!
        """
        # Save previous sequential history:
        user_seq_history_previous = {u: list(v) for u, v in self.seq_history.items()} #deepcopy(self.seq_history)
        # Update sequential history for all users with new interactions:
        update_user_seq_history(
            data[[USER_ID, ITEM_ID, POSITION_ID]].to_numpy(),
            self.seq_history,
            self.seq_len,
        )
        # Get train tensor in coo format:
        delta_inds, delta_vals = calculate_delta(
            self.seq_history, user_seq_history_previous
        )
        return delta_inds, delta_vals

    def _get_local_map(self, values):
        return {v: i for i, v in enumerate(np.sort(np.unique(values)))}

    def _map_to_local(self, data: np.ndarray, mode: int) -> int:
        mapping = self._get_local_map(data[:, mode])
        data[:, mode] = [mapping[i] for i in data[:, mode]]
        return len(mapping)

    def process_data_for_calculations(self, data: np.ndarray, mode: str):
        """
        Return processed data for different update modes.
        """
        if mode == 'new_users' or mode == 'new_items':
            mode_v = 0 if mode == 'new_users' else 1
            n_new_entities = self._map_to_local(data, mode_v)
            shape = [None, None, self.seq_len]
            shape[mode_v] = n_new_entities
            shape[1 - mode_v] = (
                self.v.shape[0] if mode == 'new_users' else self.u.shape[0]
            )
            if self.att_f is None or self.att_light_mode:
                da = unfold_sparse(
                    data[:, :3], data[:, 3], 
                    tuple(shape), 
                    mode=mode_v
                ).tocsr()
            else:
                # Incorporate Attention into new users' embs calculations: 
                da = tensordot(
                    data[:, :3], data[:, 3], 
                    tuple(shape), 
                    self.attention_mtx.T, 
                    mode=2
                )
                da = unfold_dense(da, mode=mode_v)
        elif mode == 'old':
            da = data
        elif mode == 'new':
            self._map_to_local(data, 0)
            self._map_to_local(data, 1)
            da = data
        else:
            RuntimeError(f"Bad 'mode' = {mode}!")
        return da

    def _add_zero_embeddings(self, n_new_rows, mode: str) -> None:
        if mode == 'new_users':
            _factors = self.u 
        else:
            _factors = self.v
        n_rows, n_cols = _factors.shape
        temp_emb = np.zeros(shape=(n_rows + n_new_rows, n_cols))
        temp_emb[:n_rows, :] = _factors
        if mode == 'new_users':
            self.u = temp_emb
        else: 
            self.v = temp_emb

    def _add_random_embeddings(self, n_new_rows, mode: str) -> None:
        random_state = np.random.RandomState(self.seed)
        if mode == 'new_users':
            _factors = self.u 
        else:
            _factors = self.v
        n_rows, n_cols = _factors.shape
        temp_emb = random_state.normal(
            loc=0.0, scale=self._init_new_embeddings, size=(n_rows + n_new_rows, n_cols)
        )
        temp_emb[:n_rows, :] = _factors
        temp_emb, _r = np.linalg.qr(temp_emb, mode='reduced')
        if mode == 'new_users':
            self.u = temp_emb
            unf_core = unfold_dense(self.core, 0)
            unf_core = _r.dot(unf_core)
            self.core = fold_dense(unf_core, self.core.shape, 0)
        else: 
            self.v = temp_emb
            unf_core = unfold_dense(self.core, 1)
            unf_core = _r.dot(unf_core)
            self.core = fold_dense(unf_core, self.core.shape, 1)

    def _updater(self, da, mode: str):
        """
        Internal function to update users' and items' factors.
        """
        if mode == 'new_users' or mode == 'new_items':
            if self._init_new_embeddings == 0:
                self._add_zero_embeddings(da.shape[0], mode)
            else:
                self._add_random_embeddings(da.shape[0], mode)
        elif mode == 'old':
            att_mtx = (
                self.attention_mtx 
                if (self.att_f is not None) and (not self.att_light_mode) else None 
            )
            self.u, self.v, self.w, self.core = tucker_intergrator(
                (self.u, self.v, self.w), self.core,
                da[:, :3], da[:, 3],
                shape=(self.u.shape[0], self.v.shape[0], self.seq_len), 
                parallel=self.parallel,
                att_mtx=att_mtx,
            )
        elif mode == 'new': 
            n_new_users = len(np.unique(da[:, 0]))
            n_new_items = len(np.unique(da[:, 1]))
            new_shape = (n_new_users, n_new_items, self.seq_len)
            att_mtx = (
                self.attention_mtx 
                if (self.att_f is not None) and (not self.att_light_mode) else None 
            )
            self.u, self.v, self.w, self.core = update_tucker_new_matrix(
                (self.u, self.v, self.w), self.core,
                da[:, :3], da[:, 3],
                new_shape=new_shape, 
                n_power_iter=self.n_power_iter, 
                oversampling_factor=self.oversampling_factor ,
                seed=self.seed,
                att_mtx=att_mtx,
            )
        else:
            RuntimeError(f"Bad 'mode' = {mode}!")

    def update_factors_by_mode(self, data: np.ndarray, mode: str):
        """ 
        Update user or item factors using new data.
        The function implements different modes:
            - 'new_users' - add new users
            - 'new_items' - add new items
            - 'new' - add new users that interacted only with new items; 
                (NOTE: THIS CAN WORK BADLY IF #NEW_USER of #NEW_ITEMS IS SMALL!!!)
            - 'old' - update factors for known users and items;
        """
        if len(data) > 0:
            da = self.process_data_for_calculations(data, mode)
            self._updater(da, mode=mode)

    def update(self, data: pd.DataFrame, sort_bool: bool = True) -> None:
        """
        Update user/items factors using 4-step process. Adds new users and items 
        and their corresponding embeddings if any.

        Parameters
        ----------
        data : pandas.Dataframe
            TODO
        sort_bool : bool, default: True
            TODO
        
        Returns
        -------
            None
        """
        if not self._nu_ni_process_integrator:
            # Update mappings with new entities, map data columns into internal repr:
            chunk, new_users, new_items = self.preprocess_raw_input_data(data, sort_bool)
            # Get train tensor for further computations:
            delta_inds, delta_vals = self.get_train_tensor(chunk)
            # Update factors:
            left_data = np.concatenate((delta_inds, delta_vals[:, np.newaxis]), axis=1)
            modes = ['new', 'new_users', 'new_items']
            for mode in modes:
                prepared_data, mask, new_users, new_items = _get_divide_inds_array_mask(
                    left_data, mode, new_users, new_items
                )
                self.update_factors_by_mode(prepared_data, mode=mode)
                left_data = left_data[~mask]

            # Update factors:
            left_data = np.concatenate((delta_inds, delta_vals[:, np.newaxis]), axis=1)  
            prepared_data, mask, new_users, new_items = _get_divide_inds_array_mask(
                left_data, 'new', new_users, new_items
            )
            left_data = left_data[~mask]
            self.update_factors_by_mode(left_data, mode='old')
        else:
            ############################# TEST #############################
            # Update mappings with new entities, map data columns into internal repr:
            chunk, new_users, new_items = self.preprocess_raw_input_data(data, sort_bool)
            # Get train tensor for further computations:
            delta_inds, delta_vals = self.get_train_tensor(chunk)
            # Update factors:
            left_data = np.concatenate((delta_inds, delta_vals[:, np.newaxis]), axis=1)
            
            # Add new users zero embeddings:
            if self._init_new_embeddings == 0:
                self._add_zero_embeddings(len(new_users), 'new_users')
            else:
                self._add_random_embeddings(len(new_users), 'new_users')
            # Add new items zero embeddings:
            if self._init_new_embeddings == 0:
                self._add_zero_embeddings(len(new_items), 'new_items')
            else:
                self._add_random_embeddings(len(new_items), 'new_items')

            # Apply Tucker integrator to all the data:
            self.update_factors_by_mode(left_data, mode='old')
            ############################# TEST #############################

        # Update history for filtering user's seen items in recommendations:
        self._update_history(chunk)
        # The model needs attention to do recommendations; update:
        self.aw = self.attention_mtx.dot(self.w)
        self.last_pos_emb = self.attention_mtx_inv.T.dot(self.w)[-1]

def _get_divide_inds_array_mask(
    inds: np.ndarray, 
    by: str, 
    new_users: list[int], 
    new_items: list[int]
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """
    Return masked rows of input 'inds' ndarray, the corresponding mask,
    updated lists of users and items.
    """
    old_users_mask = ~np.isin(inds[:, 0], new_users)
    old_items_mask = ~np.isin(inds[:, 1], new_items)
    if by == 'new_users':
        mask = (~old_users_mask) & (old_items_mask)
        masked_inds = inds[mask]
        processed_users = set(np.unique(masked_inds[:, 0]))
        new_users = [x for x in new_users if x not in processed_users]
    elif by == 'new_items':
        mask = (old_users_mask) & (~old_items_mask)
        masked_inds = inds[mask]
        processed_items = set(np.unique(masked_inds[:, 1]))
        new_items = [x for x in new_items if x not in processed_items]
    elif by == 'old':
        mask = (old_users_mask) & (old_items_mask)
        masked_inds = inds[mask]
    elif by == 'new':
        mask = (~old_users_mask) & (~old_items_mask)
        masked_inds = inds[mask]

        processed_users = set(np.unique(masked_inds[:, 0]))
        new_users = [x for x in new_users if x not in processed_users]
        processed_items = set(np.unique(masked_inds[:, 1]))
        new_items = [x for x in new_items if x not in processed_items]
    else:
        RuntimeError(f"Bad 'by' = {by}!")
    return masked_inds, mask, new_users, new_items
