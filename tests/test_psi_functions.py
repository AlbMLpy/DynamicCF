import sys
from typing import List

import pytest

import numpy as np
from scipy.linalg import block_diag

sys.path.append('./')
from source.psi import psi_step, update_svd_new_submatrix, update_svd_new_vectors
from source.matrix_operations import generate_matrix, split_matrix, sqrt_err_relative_mtx
from source.random_svd import random_svd, svd2mtx
from source.psi_experiments.psi_step_experiments import (
    SVD_PARAMS,
    svd_step,
    _get_experimental_results,
    NumericalExperiment,
)

RANDOM_SVD_PARAMS = {
    'n_power_iter': 1,
    'oversampling_factor': 2,
    'seed': 13,
}

def get_incremental_svd_triple(step, axis, u, s, vt, chunk, rank):
    if step == 0:
        factors_triple = random_svd(chunk, rank, **RANDOM_SVD_PARAMS)
    else:
        factors_triple = u, s, vt
        if axis == 0:
            factors_triple = (factors_triple[2].T, factors_triple[1], factors_triple[0].T)
            chunk = chunk.T
        factors_triple = update_svd_new_vectors(*factors_triple, chunk, **RANDOM_SVD_PARAMS)
        if axis == 0:
            factors_triple = (factors_triple[2].T, factors_triple[1], factors_triple[0].T)
    return factors_triple

@pytest.mark.parametrize(
    "size,rank,ratio_first,ratio_next,axis", 
    [
        (50, 10, 0.75, 1.0, 0), # one delta, new rows
        (50, 10, 0.75, 1.0, 1), # one delta, new columns
        (50, 10, 0.75, 1/5, 0), # 5 deltas, new rows
        (50, 10, 0.75, 1/5, 1), # 5 deltas, new columns
    ]
)
def test_update_svd_new_vectors_rows(size, rank, ratio_first, ratio_next, axis):
    eps = 1e-8
    err_dynamics = []
    shape_dynamics = []
    # Construct target dense matrix and split into chunks:
    c = generate_matrix((size, size), rank)
    splitted_c = split_matrix(c, ratio_first=ratio_first, ratio_next=ratio_next, axis=axis)
    factors_triple = (None, None, None)
    # Calculate computational time and error of approximations:
    for i, chunk in enumerate(splitted_c):
        target_matrix = chunk if i == 0 else np.concatenate([target_matrix, chunk], axis=axis)
        factors_triple = get_incremental_svd_triple(
            i, axis, *factors_triple, chunk, rank
        )
        err_dynamics.append(sqrt_err_relative_mtx(svd2mtx(*factors_triple), target_matrix))
        shape_dynamics.append(target_matrix.shape)
        # Check shapes of factors:
        target_factor = factors_triple[0] if axis == 0 else factors_triple[2].T
        
        expected = target_matrix.shape[axis]
        actual = target_factor.shape[0]
        assert expected == actual, f"Shape of factor = {actual.shape} != shape of target matrix = {expected.shape}"

    # Check approximation error of method:
    expected = eps
    actual = np.mean(err_dynamics)
    assert actual < eps, f"Average approximation error = {actual} > tolerance = {expected}"

def test_update_svd_new_vectors_rows_1i():
    # Prepare input:
    u = np.random.randn(20, 10)
    s = np.random.randn(10, 10)
    vt = np.random.randn(10, 15)
    new_vectors = np.zeros((20, 1))
    new_vectors[0, 0] = 1.0
    seed = 13
    _u, _, _vt = update_svd_new_vectors(u, s, vt, new_vectors, seed=seed)
    # Check shapes of factors:        
    expected = (10, 16)
    actual = _vt.shape
    assert expected == actual, f"Shape of factor = {actual.shape} != shape of target matrix = {expected.shape}"

def split_mtx_block_diag(
    mtx: np.ndarray,
    ratio_first: float,
    ratio_next: int,
) -> List[np.ndarray]:
    _, ndim = mtx.shape
    ind_first = int(ndim * ratio_first)
    step = int((ndim - ind_first) * ratio_next)

    i = 0
    dynamics = [mtx[i:i + ind_first, i:i + ind_first], ]
    i += ind_first
    n_steps = (ndim - ind_first) // step
    for _ in range(n_steps):
        dynamics.append(mtx[i:i + step, i:i + step])
        i += step
    return dynamics

def get_incremental_svd_triple_block(step, u, s, vt, chunk, rank):
    if step == 0:
        factors_triple = random_svd(chunk, rank, **RANDOM_SVD_PARAMS)
    else:
        factors_triple = u, s, vt
        factors_triple = update_svd_new_submatrix(*factors_triple, chunk)
    return factors_triple

@pytest.mark.parametrize(
    "size,rank,ratio_first,ratio_next", 
    [
        (50, 10, 0.75, 1.0), # one delta
        (50, 10, 0.75, 1/5), # 5 deltas
    ]
)
def test_update_svd_new_submatrix(size, rank, ratio_first, ratio_next):
    eps = 1e-8
    err_dynamics = []
    shape_dynamics = []
    # Construct target dense matrix and split into chunks:
    small_rank = max(int(rank // (1 / ratio_next + 1)), 1)
    c = generate_matrix((size, size), small_rank)
    splitted_c = split_mtx_block_diag(c, ratio_first, ratio_next)
    factors_triple = (None, None, None)
    # Calculate computational time and error of approximations:
    for i, chunk in enumerate(splitted_c):
        target_matrix = chunk if i == 0 else block_diag(*splitted_c[:i+1])
        factors_triple = get_incremental_svd_triple_block(
            i, *factors_triple, chunk, rank
        )

        err_dynamics.append(sqrt_err_relative_mtx(svd2mtx(*factors_triple), target_matrix))
        shape_dynamics.append(target_matrix.shape)
        # Check shapes of factors:        
        expected = target_matrix.shape
        actual = factors_triple[0].shape[0], factors_triple[2].shape[1]
        assert expected == actual, f"Shape of factor = {actual.shape} != shape of target matrix = {expected.shape}"

    # Check approximation error of method:
    expected = eps
    actual = np.mean(err_dynamics)
    assert actual < eps, f"Average approximation error = {actual} > tolerance = {expected}"

def test_update_svd_new_submatrix_1u_1i():
    # Prepare input:
    u = np.random.randn(20, 10)
    s = np.random.randn(10, 10)
    vt = np.random.randn(10, 15)
    new_rows_cols_mtx = np.array([[1.0]])
    seed = 13
    _u, _, _vt = update_svd_new_submatrix(u, s, vt, new_rows_cols_mtx, seed=seed)

    # Check shapes of factors:        
    expected = (21, 16)
    actual = (_u.shape[0], _vt.shape[1])
    assert expected == actual, f"Shape of factor = {actual.shape} != shape of target matrix = {expected.shape}"


def get_psi_dynamics(dynamics_list: List[np.ndarray], rank: int = 20):
    err_dynamics = []
    timestamps = []
    for i, a_i in enumerate(dynamics_list):
        if i == 0:
            a_i0 = a_i
            factors_triple = svd_step(a_i0, rank=rank, **SVD_PARAMS)
            err_dynamics.append(sqrt_err_relative_mtx(svd2mtx(*factors_triple), a_i0))
            timestamps.append(i)
            factors_triple = (factors_triple[0], np.diag(factors_triple[1]), factors_triple[2])
        else:
            da = a_i - a_i0
            factors_triple = psi_step(*factors_triple, da)
            u, s, vt = factors_triple
            err_dynamics.append(sqrt_err_relative_mtx(u.dot(s.dot(vt)), a_i))
            timestamps.append(i)
            a_i0 = a_i
    return _get_experimental_results([], err_dynamics, a_i0.shape[0], rank, timestamps)

@pytest.mark.parametrize(
    "n_steps,rank,step_size,eps,size,block_size", 
    [
        (20, 10, 0.1, 1e-3, 100, 10),
        (20, 10, 0.1, 1e-2, 100, 10),
        (20, 10, 0.01, 1e-2, 100, 10)
    ]
)
def test_psi_step(n_steps, rank, step_size, eps, size, block_size):
    "Test PSI against SVD. Hard test!"
    eps_tol = 1e-1
    numerical_experiment = NumericalExperiment(
        step_size=step_size, eps=eps, size=size, block_size=block_size)
    d_svd, d_psi = numerical_experiment.compare_approximations(
        n_steps=n_steps, rank=rank, plot_bool=False
    )

    expected = np.array(d_svd['error_dynamics'])
    actual = np.array(d_psi['error_dynamics'])
    assert np.abs(actual - expected).mean() < eps_tol
