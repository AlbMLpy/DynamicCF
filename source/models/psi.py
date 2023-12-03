import numpy as np
import pandas as pd

from .svd import SVD
from ..data_preparation import (
    USER_ID, 
    ITEM_ID, 
    RELEVANCE_COLUMN,
    OLD_NEW_MAP_NAME,
    update_cont_mapping_struct,
)
from ..matrix_operations import get_mapped_matrix
from ..psi import (
    update_svd_new_vectors, 
    update_svd_new_submatrix, 
    psi_step
)

class PSI(SVD):
    def __init__(
        self,
        rank,
        n_power_iter,
        oversampling_factor,
        seed,
        dtype=np.float64
    ) -> None:
        super().__init__(
            rank,
            n_power_iter,
            oversampling_factor,
            seed,
            dtype
        )
        self.seed = seed
        # Several modes for initialization and processing of new users/items:
        # - None -> use BRAND Algorithm
        # - 0 -> use zero embeddings as initialization (update is done by PSI)
        # - float x > 0 -> use random initialization (update is done by PSI)
        self._init_new_embeddings = None 
        self._nu_ni_process_integrator = False

    def _prepare_vars_new_vectors(self, mode: str):
        target_column = USER_ID if mode == 'new_users' else ITEM_ID
        other_column = ITEM_ID if mode == 'new_users' else USER_ID
        n_rows = self.vt.shape[1] if mode == 'new_users' else self.u.shape[0]
        return target_column, other_column, n_rows

    def _prepare_factors_new_vectors(self, u, s, vt, mode: str):
        self.u, self.s, self.vt = (vt.T, s, u.T) if mode == 'new_users' else (u, s, vt)

    def _add_new_entities_and_get_local_map(self, data: pd.DataFrame, col_name: str):
        new_entities = data[col_name].unique()
        update_cont_mapping_struct(self.mappings[col_name], new_entities)
        return {v: i for i, v in enumerate(new_entities)}

    def process_data_for_calculations(self, data: pd.DataFrame, mode: str):
        if (mode == 'new_users') or (mode == 'new_items'):
            column_name, row_name, n_rows = self._prepare_vars_new_vectors(mode)
            col_map = self._add_new_entities_and_get_local_map(data, column_name)
            row_map = self.mappings[row_name][OLD_NEW_MAP_NAME]
            shape = (n_rows, len(col_map))
            mtx_format = 'csc'

        elif mode == 'old':
            row_name, column_name = USER_ID, ITEM_ID
            col_map = self.mappings[column_name][OLD_NEW_MAP_NAME]
            row_map = self.mappings[row_name][OLD_NEW_MAP_NAME]
            shape = (self.u.shape[0], self.vt.shape[1])
            mtx_format = 'csr'

        elif mode == 'new':
            row_name, column_name = USER_ID, ITEM_ID
            col_map = self._add_new_entities_and_get_local_map(data, column_name)
            row_map = self._add_new_entities_and_get_local_map(data, row_name)
            shape = (len(row_map), len(col_map))
            mtx_format = 'dense'
        else:
            RuntimeError(f"Bad 'mode' = {mode}!")

        rows = data[row_name].apply(lambda x: row_map[x])
        cols = data[column_name].apply(lambda x: col_map[x])
        da = get_mapped_matrix(data[RELEVANCE_COLUMN], rows, cols, shape, mtx_format)
        return da

    def _update_history(self, data: pd.DataFrame):
        modes = (USER_ID, ITEM_ID)
        temp_history = []
        for i in range(len(self.history)):
            temp_history.append(
                np.concatenate(
                (
                    self.history[i], 
                    data[modes[i]].apply(lambda x: self.mappings[modes[i]][OLD_NEW_MAP_NAME][x])
                )
            )
        )
        self.history = (temp_history[0], temp_history[1])

    def prepare_data_with_mask_by_mode(self, data: pd.DataFrame, mode: str):
        mask = _get_divide_df_mask(data, self.mappings, mode)
        masked_data = data[mask]
        return masked_data, mask

    def _add_zero_embeddings(self, n_new_rows, mode: str) -> None:
        if mode == 'new_users':
            _factors = self.u 
        else:
            _factors = self.vt.T
        n_rows, n_cols = _factors.shape
        temp_emb = np.zeros(shape=(n_rows + n_new_rows, n_cols))
        temp_emb[:n_rows, :] = _factors
        if mode == 'new_users':
            self.u = temp_emb
        else: 
            self.vt = temp_emb.T

    def _add_random_embeddings(self, n_new_rows, mode: str) -> None:
        random_state = np.random.RandomState(self.seed)
        if mode == 'new_users':
            _factors = self.u 
        else:
            _factors = self.vt.T
        n_rows, n_cols = _factors.shape
        temp_emb = random_state.normal(
            loc=0.0, scale=self._init_new_embeddings, size=(n_rows + n_new_rows, n_cols)
        )
        temp_emb[:n_rows, :] = _factors
        temp_emb, _r = np.linalg.qr(temp_emb, mode='reduced')
        if mode == 'new_users':
            self.u = temp_emb
            self.s = np.diag(self.s) if self.s.ndim == 1 else self.s
            self.s = _r.dot(self.s)
        else: 
            self.vt = temp_emb.T
            self.s = np.diag(self.s) if self.s.ndim == 1 else self.s
            self.s =_r.dot(self.s)

    def _updater(self, da, mode: str):
        if mode == 'new_users':
            if self._init_new_embeddings is None:
                _u, _s, _vt = update_svd_new_vectors(
                    self.vt.T, self.s, self.u.T, da, *self.rsvd_params
                )
                self._prepare_factors_new_vectors(_u, _s, _vt, mode)
            elif self._init_new_embeddings == 0:
                self._add_zero_embeddings(da.shape[1], mode)
            else:
                self._add_random_embeddings(da.shape[1], mode)
        elif mode == 'new_items':
            if self._init_new_embeddings is None:
                _u, _s, _vt = update_svd_new_vectors(
                    self.u, self.s, self.vt, da, *self.rsvd_params
                )
                self._prepare_factors_new_vectors(_u, _s, _vt, mode)
            elif self._init_new_embeddings == 0:
                self._add_zero_embeddings(da.shape[1], mode)
            else:
                self._add_random_embeddings(da.shape[1], mode)
        elif mode == 'old':
            self.u, self.s, self.vt = psi_step(self.u, self.s, self.vt, da)
        elif mode == 'new':
            if self._nu_ni_process_integrator:
                self._add_zero_embeddings(da.shape[0], 'new_users')
                self._add_zero_embeddings(da.shape[1], 'new_items')
            else:
                self.u, self.s, self.vt = update_svd_new_submatrix(
                    self.u, self.s, self.vt, da, *self.rsvd_params
                )
        else:
            RuntimeError(f"Bad 'mode' = {mode}!")

    def update_factors_by_mode(self, data: pd.DataFrame, mode: str):
        if len(data) > 0:
            da = self.process_data_for_calculations(data, mode)
            self._updater(da, mode=mode)

    def update(self, data: pd.DataFrame) -> None:
        modes = ['new', 'new_users', 'new_items', 'old']
        #left_data = data
        #for mode in modes:
        #    prepared_data, mask = self.prepare_data_with_mask_by_mode(left_data, mode=mode)
        #    self.update_factors_by_mode(prepared_data, mode=mode)
        #    left_data = left_data[~mask]
        if (self._init_new_embeddings is None) and (self._nu_ni_process_integrator == False):
            left_data = data
            for mode in modes:
                prepared_data, mask = self.prepare_data_with_mask_by_mode(left_data, mode=mode)
                self.update_factors_by_mode(prepared_data, mode=mode)
                left_data = left_data[~mask]
        else:
            if (self._init_new_embeddings is not None) and (self._nu_ni_process_integrator == False):
                left_data = data
                prepared_data, mask = self.prepare_data_with_mask_by_mode(left_data, mode='new')
                self.update_factors_by_mode(prepared_data, mode='new')
                left_data = left_data[~mask]
                data_integrator = left_data.copy()
                for mode in ['new_users', 'new_items']:
                    prepared_data, mask = self.prepare_data_with_mask_by_mode(left_data, mode=mode)
                    self.update_factors_by_mode(prepared_data, mode=mode)
                    left_data = left_data[~mask]
                self.update_factors_by_mode(data_integrator, mode='old')
            elif (self._init_new_embeddings is not None) and (self._nu_ni_process_integrator == True):
                left_data = data
                for mode in ['new', 'new_users', 'new_items']:
                    prepared_data, mask = self.prepare_data_with_mask_by_mode(left_data, mode=mode)
                    self.update_factors_by_mode(prepared_data, mode=mode)
                    left_data = left_data[~mask]
                self.update_factors_by_mode(data, mode='old')
            else:
                raise RuntimeError('err')
        
        self._update_history(data)
        self.user_factors = self.u = np.array(self.u)
        self.s = np.array(self.s)
        self.s = np.diag(self.s) if self.s.ndim == 1 else self.s
        self.vt = np.array(self.vt)
        self.item_factors = self.s @ self.vt


def _get_divide_df_mask(df: pd.DataFrame, mappings, by: str):
    users = df[USER_ID].apply(lambda x: mappings[USER_ID][OLD_NEW_MAP_NAME].get(x, -1))
    items = df[ITEM_ID].apply(lambda x: mappings[ITEM_ID][OLD_NEW_MAP_NAME].get(x, -1))
    old_users_mask = ~(users == -1)
    old_items_mask = ~(items == -1)
    if by == 'new_users':
        mask = (~old_users_mask) & (old_items_mask)
    elif by == 'new_items':
        mask = (old_users_mask) & (~old_items_mask)
    elif by == 'old':
        mask = (old_users_mask) & (old_items_mask)
    elif by == 'new':
        mask = (~old_users_mask) & (~old_items_mask)
    else:
        RuntimeError(f"Bad 'by' = {by}!")
    return mask
