from typing import Tuple, Optional, Union

import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix

from .random_svd import random_svd

ORTH_EPS = 1e-10

def psi_step(
    u: np.ndarray,
    s: np.ndarray,
    vt: np.ndarray,
    da: Union[csr_matrix, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Projector Splitting Integrator(PSI) of a matrix.
    Incrementally updates SVD factors using new data of the same shape as
    factors. 

    Reference: TODO

    Parameters
    ----------
    u : numpy.ndarray[:, :]
        Left singular vectors.
    s : numpy.ndarray[:] | numpy.ndarray[:, :]
        Either singular values as a 1d array or general matrix.
    vt : numpy.ndarray[:, :]
        Right singular vectors.
    da : numpy.ndarray[:, :]
        New data to update SVD factors. da.shape == (u.shape[0], vt.shape[1]).

    Returns
    -------
    output : (numpy.ndarray[:, :], numpy.ndarray[:, :], numpy.ndarray[:, :]  
        Updated singular vectors u and vt, and "singular values" s as a general matrix.

    """
    _s = np.diag(s) if s.ndim == 1 else s

    da_v = da.dot(vt.T)
    # step 1:
    k1 = u.dot(_s) + da_v
    u1, s1 = np.linalg.qr(k1)
    # step 2:
    s0 = s1 - u1.T.dot(da_v)
    # step 3:
    l1 = vt.T.dot(s0.T) + da.T.dot(u1)
    v1, s1t = np.linalg.qr(l1)
    return u1, s1t.T, v1.T

def update_svd_new_submatrix(
    u: np.ndarray,
    s: np.ndarray,
    vt: np.ndarray,
    new_rows_cols_mtx: np.ndarray,
    n_power_iter: int = 1,
    oversampling_factor: int = 2,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Incremental update of SVD factors. Both new rows and new columns.

    Reference: TODO

    Parameters
    ----------
    u : numpy.ndarray[:, :]
        Left singular vectors.
    s : numpy.ndarray[:] | numpy.ndarray[:, :]
        Either singular values as a 1d array or general matrix.
    vt : numpy.ndarray[:, :]
        Right singular vectors.
    new_rows_cols_mtx : numpy.ndarray[:, :]
        New rows and columns matrix.
    n_power_iter : int, default: 1
        TODO
    oversampling_factor : int, default: 2
        TODO
    seed : int, optional, default: None
        TODO

    Returns
    -------
    output : (numpy.ndarray[:, :], numpy.ndarray[:], numpy.ndarray[:, :]  
        Updated singular vectors u and vt, and singular values s.

    """
    _, rank = u.shape
    u1, s1, v1t = np.linalg.svd(new_rows_cols_mtx, full_matrices=False)
    u2 = block_diag(u, u1)
    v2 = block_diag(vt.T, v1t.T)
    if s.ndim == 1:
        s2 = np.concatenate((s, s1), axis=0)
        mask = np.argsort(s2)[::-1][:rank]
        return u2[:, mask], s2[mask], v2[:, mask].T
    else:
        s2 = block_diag(s, np.diag(s1))
        _u, _s, _vt = random_svd(
            s2, rank, n_power_iter, oversampling_factor, seed=seed
        )
        u2 = u2.dot(_u)
        v2t = v2.dot(_vt.T).T
        return u2, _s, v2t

def update_svd_new_vectors(
    u: np.ndarray,
    s: np.ndarray, 
    vt: np.ndarray,
    new_vectors: np.ndarray,
    n_power_iter: int = 1,
    oversampling_factor: int = 2,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Incremental update of SVD factors. Only new rows or new columns.

    Reference: TODO

    Parameters
    ----------
    u : numpy.ndarray[:, :]
        Left singular vectors.
    s : numpy.ndarray[:] | numpy.ndarray[:, :]
        Either singular values as a 1d array or general matrix.
    vt : numpy.ndarray[:, :]
        Right singular vectors.
    new_vectors : numpy.ndarray[:, :]
        New vectors to add as columns.
    n_power_iter : int, default: 1
        TODO
    oversampling_factor : int, default: 2
        TODO
    seed : int, optional, default: None
        TODO

    Returns
    -------
    output : (numpy.ndarray[:, :], numpy.ndarray[:], numpy.ndarray[:, :]  
        Updated singular vectors u and vt, and singular values s.

    Examples:
        - add more columns:
            u1, s1, v1t = random_svd(old_column_vectors, RANK, N_POWER_ITER, OVERSAMPLING_FACTOR, seed=SEED)
            u2, s2, v2t = update_svd_new_vectors(
                u1, s1, v1t, new_column_vectors, N_POWER_ITER, OVERSAMPLING_FACTOR, seed=SEED
            )
        - add more rows:
            u1, s1, v1t = random_svd(old_row_vectors, RANK, N_POWER_ITER, OVERSAMPLING_FACTOR, seed=SEED)
            u2, s2, v2t = update_svd_new_vectors(
                v1t.T, s1, u1.T, new_row_vectors.T, N_POWER_ITER, OVERSAMPLING_FACTOR, seed=SEED
            )
    """
    _, rank = u.shape
    _, q = vt.shape
    _, nc = new_vectors.shape

    _s = np.diag(s) if s.ndim == 1 else s
    
    projection = u.T @ new_vectors
    complement = new_vectors - u.dot(projection)
    j, k = np.linalg.qr(complement)
    j = np.array(j)
    
    uj = np.concatenate([u, j], axis=1)

    vi = np.zeros((q + nc, rank + nc))
    vi[:q, :rank] = vt.T
    vi[q:, rank:] = np.identity(nc)

    dq = np.zeros((rank + nc, rank + nc))
    dq[:rank, :rank] = _s
    dq[:rank, rank:] = projection
    dq[rank:, rank:] = k

    # Reorthogonalize when the inner product of its first 
    # and last columns exceeds some small eps = ORTH_EPS:
    orth_err = uj[:, 0].dot(uj[:, -1])
    if orth_err > ORTH_EPS:
        uj, _k = np.linalg.qr(uj)
        dq = _k.dot(dq)
    
    u1, s2, vt1 = random_svd(
        dq, rank, n_power_iter, oversampling_factor, seed=seed
    )
    u2 = uj.dot(u1)
    v2t = vi.dot(vt1.T).T
    return u2, s2, v2t
