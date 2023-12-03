from typing import Tuple, Optional

import numpy as np
from scipy.linalg import block_diag
from scipy.sparse.linalg import LinearOperator, svds

from source.tensor_operations import (
    unfold_dense,
    fold_dense, 
    construct_core, 
    construct_core_parallel,
    sparse_tensor_double_tensor,
    tensordot, 
    D_TYPE
)
from source.psi import update_svd_new_vectors
from source.random_svd import random_svd
from source.hosvd import hosvd_dense, tucker2_dense

N_MODES = 3
MAX_MODE_IND = N_MODES - 1

def tucker_intergrator(
    factors: Tuple[np.ndarray, np.ndarray, np.ndarray],
    core: np.ndarray, 
    inds: np.ndarray, 
    vals: np.ndarray, 
    shape: Tuple[int, int, int], 
    parallel: bool = False,
    att_mtx: Optional[np.ndarray] = None
):
    """
    Calculate Tucker Integrator of sparse tensor provided in COO format.
    Incrementally updates factors using new data of the same shape as
    factors.

    Reference: TODO

    Parameters
    ----------
    factors : tuple of numpy.ndarray
        TODO
    core : numpy.ndarray
        TODO
    inds : numpy.ndarray
        TODO
    vals : numpy.ndarray
        TODO
    shape : tuple of ints
        TODO
    parallel : bool, default: False
        TODO
    att_mtx : numpy.ndarray, optional, default: None
        Attention matrix for the third mode (RecSys - GA-SATF)

    Returns
    -------
    output : tuple of numpy.ndarray 
        TODO

    """
    c0 = 1 * core # copy core
    _, u1, u2 = factors
    new_factors = []
    core_function = construct_core_parallel if parallel else construct_core

    for i in range(N_MODES):
        #1
        c0_i = unfold_dense(c0, mode=i)
        q0_i, s0_i = np.linalg.qr(c0_i.T, mode='reduced')
        s0_i = s0_i.T
        #2
        if (att_mtx is not None) and (i < MAX_MODE_IND):
            u2 = att_mtx.dot(u2)
        da_v0_i = sparse_tensor_double_tensor(inds, vals, shape, q0_i, mode=i, u1=u1, u2=u2)
        if (att_mtx is not None) and (i == MAX_MODE_IND):
            da_v0_i = att_mtx.T.dot(da_v0_i)
        #3, 4
        k0_i = factors[i].dot(s0_i).astype(D_TYPE)
        k1_i = k0_i + da_v0_i
        #5,6
        u1_i, s0_i = np.linalg.qr(k1_i, mode='reduced')
        new_factors.append(u1_i)
        #7
        s1_i = s0_i - u1_i.T.dot(da_v0_i)
        #8
        c0_i = s1_i.dot(q0_i.T)
        c0 = fold_dense(c0_i, c0.shape, mode=i)
        ##9
        if i == 0:
            u1 = u1_i
        else:
            u2 = u1_i
    # Recalculate Tucker Core:
    if att_mtx is not None:
        factors_core = (*new_factors[:MAX_MODE_IND], att_mtx.dot(new_factors[MAX_MODE_IND]))
    else:
        factors_core = new_factors
    c1 = c0 + core_function(inds, vals, *factors_core)
    return *new_factors, c1

def update_tucker_new_vectors(
    factors: Tuple[np.ndarray, np.ndarray, np.ndarray],
    core: np.ndarray, 
    new_vectors: np.ndarray,
    mode: int,
    n_power_iter: int = 1,
    oversampling_factor: int = 2,
    seed: Optional[int] = None,
    explicit_calc: bool = True,
):
    """
    Incremental update of Tucker factors. Only new vectors per one mode are added.

    Reference: TODO

    Parameters
    ----------
    factors : tuple of numpy.ndarray
        TODO
    core : numpy.ndarray
        TODO
    new_vectors : numpy.ndarray[:, :]
        New vectors to add per mode
    mode : int
        TODO
    n_power_iter : int, default: 1
        TODO
    oversampling_factor : int, default: 2
        TODO
    seed : int, optional, default: None
        TODO
    explicit_calc : bool, default: True
        Use naive implementation to calculate embeddings or not.

    Returns
    -------
    output : tuple of numpy.ndarray   
        Updated factors and core.

    Examples:
        - add new factor vectors per mode 0:
        mode = 0
        new_entities = unfold_sparse(
            initial_data[[USER_ID, ITEM_ID, 'order']].to_numpy(), initial_data['relevance'].values, (m, n ,l), mode
        )
        *nf, nc = update_tucker_new_vectors(factors, core, new_entities.tocsr(), mode, seed=13)
        - add new factor vectors per mode 1:
        mode = 1
        new_entities = unfold_sparse(
            initial_data[[USER_ID, ITEM_ID, 'order']].to_numpy(), initial_data['relevance'].values, (m, n ,l), mode
        )
        *nf, nc = update_tucker_new_vectors(factors, core, new_entities.tocsr(), mode, seed=13)

    """
    modes = np.arange(3)
    mode_a, mode_b = modes[modes != mode]
    ranks = core.shape
    # Preprocess factors, core:
    core_unf = unfold_dense(core, mode)
    w_v = np.kron(factors[mode_b].T, factors[mode_a].T)
    # Choose mode:
    if explicit_calc:
        x = core_unf.dot(w_v)
        # x - short & wide dense matrix - FASTER VERSION!!!: -> BOTTLENECK!!!!!!!!!!!!!!!!!!!!!!
        v1t, s1, u1 = random_svd(x.T, ranks[mode], n_power_iter, oversampling_factor, seed=seed)
        u1 = u1.T
        v1t = v1t.T
    else:
        lf, rf = factors[mode_b], factors[mode_a] # W \kron V
        l_shape = (rf.shape[1], lf.shape[1])
        r_shape = (rf.shape[0], lf.shape[0])

        def kron_mv(v):
            cv = core_unf.T.dot(v)
            cv_mat = cv.reshape(l_shape, order='F')
            res = np.linalg.multi_dot([rf, cv_mat, lf.T])
            return res.flatten(order='F')

        def kron_rmv(v):
            v_mat = v.reshape(r_shape, order='F')
            rvl = np.linalg.multi_dot([rf.T, v_mat, lf])
            rvl_vec = rvl.flatten(order='F')
            return core_unf.dot(rvl_vec)

        mtx_shape = (r_shape[0]*r_shape[1], ranks[mode])
        shifted_matrix = LinearOperator(
            mtx_shape,
            kron_mv,
            kron_rmv
        )
        v1t, s1, u1 = svds(shifted_matrix, k=ranks[mode], solver='propack')
        v1t = np.ascontiguousarray(v1t[:, ::-1]).T
        s1 = s1[::-1]
        u1 = np.ascontiguousarray(u1[::-1, :]).T

    # Add new vectors:
    u1 = factors[mode].dot(u1)
    v1t, s1, u1 = update_svd_new_vectors(v1t.T, s1, u1.T, new_vectors.T, seed=seed)
    u1, v1t = u1.T, v1t.T
    s_vt = s1[:, np.newaxis] * v1t
    # Update core:
    core = s_vt.dot(w_v.T)
    core = fold_dense(core, ranks, mode)
    # Update factors:
    factors_upd = [None, None, None]
    factors_upd[mode] = u1
    factors_upd[mode_a] = factors[mode_a]
    factors_upd[mode_b] = factors[mode_b]
    return *factors_upd, core

def update_tucker_new_matrix(
    factors: Tuple[np.ndarray, np.ndarray, np.ndarray],
    core: np.ndarray, 
    new_inds: np.ndarray,
    new_vals: np.ndarray,
    new_shape: Tuple[int, int, int],
    n_power_iter: int = 1,
    oversampling_factor: int = 2,
    seed: Optional[int] = None,
    att_mtx: Optional[np.ndarray] = None,
):
    """
    Incremental update of Tucker factors. 
    Only new vectors per 0 mode and new vectors per 1 mode are added.

    Reference: TODO

    Parameters
    ----------
    factors : tuple of numpy.ndarray
        TODO
    core : numpy.ndarray
        TODO
    new_inds : numpy.ndarray
        TODO
    new_vals : numpy.ndarray
        TODO
    new_shape : tuple of int
        TODO 
    n_power_iter : int, default: 1
        TODO
    oversampling_factor : int, default: 2
        TODO
    seed : int, optional, default: None
        TODO
    att_mtx : numpy.ndarray, optional, default: None
        Attention matrix for the third mode (RecSys - GA-SATF)

    Returns
    -------
    output : tuple of numpy.ndarray   
        Updated factors and core.

    Examples:
        TODO

    """
    u0, v0, w0 = factors
    if att_mtx is not None:
        aw = att_mtx.dot(w0)
    else:
        aw = w0
    x = tensordot(new_inds, new_vals, new_shape, aw.T, mode=2)
    # Tucker decomposition of new data chunk:
    _u, _v, _core = tucker2_dense(
        x, rank=x.shape,
        identity_mode=2, n_power_iter=n_power_iter,
        oversampling_factor=oversampling_factor,
        seed=seed,
    )
    # Prepare extended core:
    c0, c1, _ = core.shape
    nc0, nc1, _ = _core.shape
    _core_1 = np.zeros((c0 + nc0, c1 + nc1, core.shape[2]))
    _core_1[:c0, :c1, :] = core
    _core_1[c0:, c1:, :] = _core
    # Tucker decomposition of new core:
    u1, v1, w1, new_core = hosvd_dense(_core_1, core.shape, seed=seed)
    # Update factors:
    u = block_diag(u0, _u)
    u = u.dot(u1)

    v = block_diag(v0, _v)
    v = v.dot(v1)

    w = w0.dot(w1)
    return u, v, w, new_core
