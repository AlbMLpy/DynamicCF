from typing import Tuple, List, Optional

import numpy as np
from numba import jit
from scipy.sparse import coo_matrix

from .random_svd import svd2mtx

D_TYPE = np.float64
EPS = 1e-14
NO_PYTHON=True


@jit(nopython=NO_PYTHON)
def get_svd_element(
    i: int,
    j: int,
    u: np.ndarray,
    s: np.ndarray,
    vt: np.ndarray,
) -> float:
    u_i = u[i]
    v_j = vt.T[j]
    _, rank = u.shape
    res = 0.0
    for r in range(rank):
        res += u_i[r] * s[r] * v_j[r]
    return res

@jit(nopython=NO_PYTHON)
def get_mf_element(
    i: int,
    j: int,
    u: np.ndarray,
    vt: np.ndarray,
) -> float:
    u_i = u[i]
    v_j = vt.T[j]
    _, rank = u.shape
    res = 0.0
    for r in range(rank):
        res += u_i[r] * v_j[r]
    return res
    
@jit(nopython=NO_PYTHON)
def _sqrt_err(
    inds: np.ndarray,
    vals: np.ndarray,
    u: np.ndarray,
    s: np.ndarray,
    vt: np.ndarray,
) -> float:
    result = 0.0
    if s.ndim == 2:
        _vt = s.dot(vt)
        for item in range(inds.shape[0]):
            i, j = inds[item]
            result += (vals[item] - get_mf_element(i, j, u, _vt))**2 
    else:
        for item in range(inds.shape[0]):
            i, j = inds[item]
            result += (vals[item] - get_svd_element(i, j, u, s, vt))**2     
    return np.sqrt(result)

@jit(nopython=NO_PYTHON)
def sqrt_err_relative(
    inds: np.ndarray,
    vals: np.ndarray,
    u: np.ndarray,
    s: np.ndarray,
    vt: np.ndarray,
) -> float:
    result = _sqrt_err(inds, vals, u, s, vt)         
    return result / np.sqrt((vals**2).sum() + EPS)

def get_mapped_matrix(vals, rows, cols, shape, mtx_format: str = 'coo', dtype = np.float64):
    a = coo_matrix((vals, (rows, cols)), shape=shape, dtype=dtype)
    if mtx_format == 'csr':
        a = a.tocsr()
    elif mtx_format == 'csc':
        a = a.tocsc()
    elif mtx_format == 'dense':
        a = a.A
    return a

##### Dense Operations #####
def generate_matrix(
    shape: Tuple[int, int],
    rank: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate random matrix with definite shape and rank.
    """
    random_state = np.random if seed is None else np.random.RandomState(seed)

    n_rows, n_cols = shape
    a = random_state.randn(n_rows, rank)
    q1, _ = np.linalg.qr(a, mode='reduced')
    b = random_state.randn(n_cols, rank)
    q2, _ = np.linalg.qr(b, mode='reduced')
    qs = random_state.randn(rank)
    mtx = svd2mtx(q1, qs, q2.T)
    return mtx

def split_matrix(
    mtx: np.ndarray,
    ratio_first: float,
    ratio_next: float,
    axis: int
) -> List[np.ndarray]:
    """
    Split matrix by rows or columns into blocks.

    Parameters
    ----------
    mtx : numpy.ndarray
        TODO
    ratio_first : float
        TODO
    ratio_next : float
        TODO
    axis : int
        TODO

    Returns
    -------
    output : sequence of numpy.ndarray 
        TODO

    """
    ndim = mtx.shape[axis]
    ind_first = int(ndim * ratio_first)
    step = int((ndim - ind_first) * ratio_next)
    return np.array_split(
        mtx,
        np.arange(ind_first, ndim, step),
        axis=axis
    )

def _mean_square(a: np.ndarray) -> float:
    return np.mean(a**2)

def sqrt_err_relative_mtx(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate relative error of matrices difference: sqrt(mse(a-b) / mse(b)).
    """
    error = a - b
    ms_error = _mean_square(error)
    denom = _mean_square(b)
    return np.sqrt(ms_error / (denom + EPS))
##### Dense Operations #####
