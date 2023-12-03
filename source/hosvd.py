from typing import Tuple, Optional

import numpy as np
from scipy.sparse.linalg import svds

from .rp_hooi import (
    N_MODES,
    check_params,
    init_factors, 
    log_status,
    tucker_rank_is_valid,
)
from .random_svd import random_svd
from .tensor_operations import (
    unfold_dense, 
    unfold_sparse, 
    construct_core, 
    construct_core_parallel,
    construct_core_parallel_dense
)

def hosvd(
    inds: np.ndarray,
    vals: np.ndarray,
    shape: Tuple[int, int, int],
    rank: Tuple[int, int, int],
    seed: Optional[int] = None,
    verbose: bool = False,
    parallel: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Tucker decomposition of sparse tensor
    provided in COO format using HOSVD algorithm with
    random initialization.

    Reference: TODO

    Parameters
    ----------
    inds : numpy.ndarray
        TODO
    vals : numpy.ndarray
        TODO
    shape : tuple of ints
        TODO
    rank : tuple of ints
        TODO
    seed : int, optional, default: None
        TODO
    verbose : bool, default: False
        TODO
    parallel : bool, default: False
        TODO

    Returns
    -------
    output : sequence of numpy.ndarray 
        TODO

    """
    check_params(shape, rank)    
    factors_list = init_factors(shape, rank, orthogonal=False, seed=seed)
    core_function = construct_core_parallel if parallel else construct_core

    log_status("Start training", verbose)
    for mode in range(N_MODES):
        log_status(f"mode -> {mode}", verbose)
        x_n = unfold_sparse(inds, vals, shape, mode)
        factors_list[mode] = svds(
            x_n,
            k=rank[mode],
            return_singular_vectors='u',
            random_state=seed,
        )[0]
        factors_list[mode] = np.ascontiguousarray(factors_list[mode][:, ::-1])
    
    core = core_function(inds, vals, *factors_list)
    return *factors_list, core

def hosvd_dense(
    x: np.ndarray,
    rank: Tuple[int, int, int],
    n_power_iter: int = 1,
    oversampling_factor: int = 2,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Tucker decomposition of sparse tensor
    provided in COO format using HOSVD algorithm with
    random initialization.

    Reference: TODO

    Parameters
    ----------
    x : numpy.ndarray
        TODO
    rank : tuple of ints
        TODO
    n_power_iter : int, default: 1
        TODO
    oversampling_factor : int, default: 2
        TODO
    seed : int, optional, default: None
        TODO
    verbose : bool, default: False
        TODO

    Returns
    -------
    output : sequence of numpy.ndarray 
        TODO

    """
    shape = x.shape
    check_params(shape, rank)    
    factors_list = init_factors(shape, rank, orthogonal=False, seed=seed)

    log_status("Start training", verbose)
    for mode in range(N_MODES):
        log_status(f"mode -> {mode}", verbose)
        x_n = unfold_dense(x, mode)
        _, _, ut = random_svd(
            x_n.T,
            rank=rank[mode],
            n_power_iter=n_power_iter,
            oversampling_factor=oversampling_factor,
            seed=seed
        )
        factors_list[mode] = ut.T
        factors_list[mode] = np.ascontiguousarray(factors_list[mode][:, ::-1])
    
    core = construct_core_parallel_dense(x, *factors_list)
    return *factors_list, core

def tucker2_dense(
    x: np.ndarray,
    rank: Tuple[int, int, int],
    identity_mode: int,
    n_power_iter: int = 1,
    oversampling_factor: int = 10,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = x.shape
    factors_list = init_factors(shape, rank, orthogonal=False, seed=seed)
    factors_list[identity_mode] = np.eye(rank[identity_mode])
    modes = np.arange(N_MODES)

    log_status("Start training", verbose)
    for mode in range(N_MODES):
        if mode != identity_mode:
            log_status(f"mode -> {mode}", verbose)
            x_n = unfold_dense(x, mode)
            if tucker_rank_is_valid(rank):
                _, _, ut = random_svd(
                    x_n.T,
                    rank=rank[mode],
                    n_power_iter=n_power_iter,
                    oversampling_factor=oversampling_factor,
                    seed=seed
                )
            else:
                _, _, ut = np.linalg.svd(x_n.T, full_matrices=True)
            factors_list[mode] = ut.T
            factors_list[mode] = np.ascontiguousarray(factors_list[mode][:, ::-1])
    
    core = construct_core_parallel_dense(x, *factors_list)
    mode_a, mode_b = modes[modes != identity_mode]
    return factors_list[mode_a], factors_list[mode_b], core
