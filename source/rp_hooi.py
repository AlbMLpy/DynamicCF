from typing import Sequence, Tuple, Optional, Callable

import numpy as np

from .tensor_operations import construct_core, construct_core_parallel, tensordot2
from .random_svd import random_svd

TuckerFactorsCore = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
TuckerFactors = tuple[np.ndarray, np.ndarray, np.ndarray]

EPS = 1e-8
N_MODES = 3

def tucker_rank_is_valid(rank: Sequence[int]) -> bool:
    modes = np.arange(len(rank))
    for mode in modes:
        mu, mv = modes[modes != mode]
        if rank[mode] > rank[mu]*rank[mv]:
            return False
    return True

def log_status(msg, verbose):
    if verbose:
        print(msg)

def check_params(shape, rank):
    if len(shape) > N_MODES:
        raise NotImplementedError("Considering only 3-way tensors")
    if not tucker_rank_is_valid(rank):
        raise RuntimeError(f"Bad rank: {rank}. Every value must be less than the product of the others")

def init_factors(shape, rank, orthogonal: bool = True, seed: Optional[int] = None):
    random_state = np.random if seed is None else np.random.RandomState(seed)
    factors_list = [None, None, None]
    for mode in range(N_MODES):
        u = random_state.rand(shape[mode], rank[mode])
        if orthogonal:
            u = np.linalg.qr(u, mode='reduced')[0]
        factors_list[mode] = u
    return factors_list

def _core_grew_bool(g_growth, g_norm_old, growth_tol, verbose) -> bool:
    log_status(f'Growth of the core: {g_growth}', verbose)
    if g_growth < growth_tol:
        log_status(f'Core is no longer growing. Norm of the core: {g_norm_old}', verbose)
        return False
    return True

def rp_hooi(
    inds: np.ndarray,
    vals: np.ndarray,
    shape: Tuple[int, int, int],
    rank: Tuple[int, int, int],
    n_iter: int = 25,
    growth_tol: float = 0.01,
    n_power_iter: int = 1,
    oversampling_factor: int = 10,
    seed: Optional[int] = None,
    verbose: bool = False,
    parallel: bool = False,
) -> TuckerFactorsCore:
    """
    Calculate Tucker decomposition of sparse tensor
    provided in COO format using HOOI algorithm
    with random initialization and random SVD.

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
    n_iter : int, default: 25
        TODO
    growth_tol : float, default: 0.01
        TODO
    n_power_iter : int, default: 1
        TODO
    oversampling_factor : int, default: 10
        TODO
    seed : int, optional, default: None
        TODO
    verbose : bool, default: False
        TODO
    parallel : bool, default: False
        TODO

    Returns
    -------
    output : tuple of numpy.ndarray 
        TODO

    """
    check_params(shape, rank)
    factors_list = init_factors(shape, rank, orthogonal=True, seed=seed)

    g_norm_old = 0
    core = np.empty(0)
    core_function = construct_core_parallel if parallel else construct_core
    modes = np.arange(N_MODES)
    for i in range(n_iter):
        log_status(f'Step {i+1} of {n_iter}', verbose)
        for mode in range(N_MODES):
            mu, mv = modes[modes != mode]
            u, v = factors_list[mu], factors_list[mv]
            factors_list[mode] = tensordot2(inds, vals, shape, u, v, (mu, mv)).reshape(shape[mode], rank[mu]*rank[mv])
            uu = random_svd(factors_list[mode], rank[mode], n_power_iter, oversampling_factor, seed=seed)[0] 
            factors_list[mode] = np.ascontiguousarray(uu[:, ::-1])

        core = core_function(inds, vals, *factors_list)
        g_norm_new = np.linalg.norm(core)
        g_growth = (g_norm_new - g_norm_old) / (g_norm_new + EPS)
        g_norm_old = g_norm_new
        if not _core_grew_bool(g_growth, g_norm_old, growth_tol, verbose):
            break    
    return *factors_list, core

def tucker2(
    inds: np.ndarray,
    vals: np.ndarray,
    shape: Tuple[int, int, int],
    rank: Tuple[int, int, int],
    identity_mode: int,
    n_iter: int = 25,
    growth_tol: float = 0.01,
    n_power_iter: int = 1,
    oversampling_factor: int = 10,
    seed: Optional[int] = None,
    verbose: bool = False,
    parallel: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    check_params(shape, rank)
    factors_list = init_factors(shape, rank, orthogonal=True, seed=seed)
    factors_list[identity_mode] = np.eye(rank[identity_mode])

    g_norm_old = 0
    core = np.empty(0)
    core_function = construct_core_parallel if parallel else construct_core
    modes = np.arange(N_MODES)
    for i in range(n_iter):
        log_status(f'Step {i+1} of {n_iter}', verbose)
        for mode in range(N_MODES):
            if mode != identity_mode:
                mu, mv = modes[modes != mode]
                u, v = factors_list[mu], factors_list[mv]
                factors_list[mode] = tensordot2(inds, vals, shape, u, v, (mu, mv)).reshape(shape[mode], rank[mu]*rank[mv])
                uu = random_svd(factors_list[mode], rank[mode], n_power_iter, oversampling_factor, seed=seed)[0] 
                factors_list[mode] = np.ascontiguousarray(uu[:, ::-1])

        core = core_function(inds, vals, *factors_list)

        g_norm_new = np.linalg.norm(core)
        g_growth = (g_norm_new - g_norm_old) / (g_norm_new + EPS)
        g_norm_old = g_norm_new
        if not _core_grew_bool(g_growth, g_norm_old, growth_tol, verbose):
            break  
    mode_a, mode_b = modes[modes != identity_mode]
    return factors_list[mode_a], factors_list[mode_b], core

def ga_satf(
    inds: np.ndarray,
    vals: np.ndarray,
    shape: Tuple[int, int, int],
    rank: Tuple[int, int, int],
    attention_mtx: np.ndarray,
    n_iter: int = 25,
    growth_tol: float = 0.01,
    n_power_iter: int = 1,
    oversampling_factor: int = 10,
    seed: Optional[int] = None,
    verbose: bool = False,
    parallel: bool = False,
    exit_callback: Optional[Callable[[TuckerFactorsCore], bool]] = None,
    init_factors_list: Optional[TuckerFactors] = None,
    force_n_iter: bool = False,
) -> TuckerFactorsCore:
    """
    Calculate Globally Attentive Sequence-Aware Tensor
    Factorization (GA-SATF) of sparse tensor
    provided in COO format using HOOI algorithm
    with random initialization and random SVD.

    Reference: https://arxiv.org/abs/2212.05720

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
    attention_mtx : numpy.ndarray
        Global attention
    n_iter : int, default: 25
        TODO
    growth_tol : float, default: 0.01
        TODO
    n_power_iter : int, default: 1
        TODO
    oversampling_factor : int, default: 10
        TODO
    seed : int, optional, default: None
        TODO
    verbose : bool, default: False
        TODO
    parallel : bool, default: False
        TODO
    exit_callback : callable, optional, default: None
        Return True if factors and core are "good", else False
    init_factors : tuple of numpy.ndarray, optional, default: None
        Factors initialization.
    force_n_iter : bool
        Use strictly n_iter cycles in a loop of a model.

    Returns
    -------
    output : tuple of numpy.ndarray 
        TODO

    """
    check_params(shape, rank)
    factors_list = init_factors(shape, rank, orthogonal=True, seed=seed)
    if init_factors_list is not None:
        for i in range(len(factors_list)):
            n_rows, _ = init_factors_list[i].shape
            factors_list[i][:n_rows] = init_factors_list[i]

    g_norm_old = 0
    core = np.empty(0)
    core_function = construct_core_parallel if parallel else construct_core
    modes = np.arange(N_MODES)
    for i in range(n_iter):
        log_status(f'Step {i+1} of {n_iter}', verbose)
        factors_list[2] = attention_mtx.dot(factors_list[2])
        for mode in range(N_MODES):
            mu, mv = modes[modes != mode]
            u, v = factors_list[mu], factors_list[mv]
            factors_list[mode] = tensordot2(inds, vals, shape, u, v, (mu, mv)).reshape(shape[mode], rank[mu]*rank[mv])
            if mode == 2:
                factors_list[mode] = attention_mtx.T.dot(factors_list[mode])
            uu = random_svd(factors_list[mode], rank[mode], n_power_iter, oversampling_factor, seed=seed)[0] 
            factors_list[mode] = np.ascontiguousarray(uu[:, ::-1])
        core = core_function(inds, vals, *factors_list[:2], attention_mtx.dot(factors_list[2]))
        if not force_n_iter:
            if exit_callback is None:
                g_norm_new = np.linalg.norm(core)
                g_growth = (g_norm_new - g_norm_old) / (g_norm_new + EPS)
                g_norm_old = g_norm_new
                if not _core_grew_bool(g_growth, g_norm_old, growth_tol, verbose):
                        break  
            else:
                if exit_callback(*factors_list, core):
                    break 
    return *factors_list, core
