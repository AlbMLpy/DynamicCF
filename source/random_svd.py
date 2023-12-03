from typing import Tuple, Optional, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

def random_svd(
    x: np.ndarray,
    rank: int,
    n_power_iter: int = 1,
    oversampling_factor: int = 2,
    full_matrices: bool = False,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Random SVD of a matrix.

    Reference: TODO

    Parameters
    ----------
    x : numpy.ndarray
        The target matrix to get an SVD from.
    rank : int
        The number of latent factors.
    n_power_iter : int, default: 1
        TODO
    oversampling_factor : int, default: 2
        TODO
    full_matrices : bool, default: False
        TODO
    seed : int, optional, default: None
        TODO

    Returns
    -------
    output : sequence of numpy.ndarray 
        TODO

    """ 
    random_state = np.random if seed is None else np.random.RandomState(seed)
    _, n_cols = x.shape
    p = random_state.randn(n_cols, rank + oversampling_factor)
    z = x.dot(p)
    for _ in range(n_power_iter):
        z = x.dot(x.T.dot(z))
    q, _ = np.linalg.qr(z, mode='reduced')

    y = q.T.dot(x)
    uy, s, vt = np.linalg.svd(y, full_matrices=full_matrices)
    u = q.dot(uy)
    return u[:, :rank], s[:rank], vt[:rank, :]

def svd_step(
    a: Union[csr_matrix, np.ndarray],
    rank: int ,
    n_power_iter: int = 1,
    oversampling_factor: int = 2,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Singular Value Decomposition of a matrix.
    Uses scipy.sparse.linalg.svds if a matrix is in 'csr' format.
    Uses random SVD if a matrix is numpy.ndarray.

    Reference: TODO

    Parameters
    ----------
    a : numpy.ndarray[:, :] | csr_matrix[:, :]
        The target matrix to get an SVD from.
    rank : int
        The number of latent factors.
    n_power_iter : int, default: 1
        TODO
    oversampling_factor : int, default: 2
        TODO
    seed : int, optional, default: None
        TODO

    Returns
    -------
    output : (numpy.ndarray[:, :], numpy.ndarray[:], numpy.ndarray[:, :]  
        TODO

    """
    if isinstance(a, csr_matrix):
        u, s, vt = svds(a, k=rank, random_state=seed)
        u = u[:, ::-1]
        s = s[::-1]
        vt = vt[::-1, :]
    elif isinstance(a, np.ndarray):
        u, s, vt = random_svd(a, rank, n_power_iter, oversampling_factor, seed=seed)
    else:
        raise ValueError("Bad type of argument 'a'")
    return u, s, vt

def svd2mtx(u: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Get matrix A = USV^T from SVD factor matrices.

    Parameters
    ----------
    u : numpy.ndarray
        TODO
    s : numpy.ndarray
        TODO
    v : numpy.ndarray
        TODO

    Returns
    -------
    output : numpy.ndarray 
        TODO
        
    """
    return (u * s).dot(v)
