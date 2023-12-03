from typing import Tuple, Union, Optional, Dict, List

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from ..random_svd import random_svd, svd2mtx
from ..general_functions import elapsed_time
from ..matrix_operations import sqrt_err_relative_mtx, generate_matrix, split_matrix
from ..plotting import PARAMS, get_fig_ax, add_plot_ax, set_ax
from ..psi import psi_step, update_svd_new_vectors, update_svd_new_submatrix

SVD_PARAMS = {
    'n_power_iter': 1,
    'oversampling_factor': 2,
    'seed': 13,
}

def _get_experimental_results(time_dynamics, err_dynamics, size, rank, shape_dynamics):
    results = {}
    results['ctime_dynamics'] = time_dynamics
    results['error_dynamics'] = err_dynamics
    results['matrix_size'] = size
    results['matrix_rank'] = rank
    results['shape_dynamics'] = shape_dynamics
    return results

######### About adding new rows/columns: #########
def get_svd_triple(step, axis, u, s, vt, target_matrix, chunk, rank, params):
    factors_triple, comp_time = elapsed_time(random_svd, target_matrix, rank, **params)
    return factors_triple, comp_time

def get_incremental_svd_triple(step, axis, u, s, vt, target_matrix, chunk, rank, params):
    if step == 0:
        factors_triple, comp_time = elapsed_time(random_svd, chunk, rank, **params)
    else:
        factors_triple = u, s, vt
        if axis == 0:
            factors_triple = (factors_triple[2].T, factors_triple[1], factors_triple[0].T)
            chunk = chunk.T
        factors_triple, comp_time = elapsed_time(update_svd_new_vectors, *factors_triple, chunk, **params)
        if axis == 0:
            factors_triple = (factors_triple[2].T, factors_triple[1], factors_triple[0].T)
    return factors_triple, comp_time

def get_dynamics_vec(size, rank, ratio_first, ratio_next, axis, triple_func, triple_func_params):
    time_dynamics = []
    err_dynamics = []
    shape_dynamics = []
    # Construct target dense matrix and split into chunks:
    c = generate_matrix((size, size), rank)
    splitted_c = split_matrix(c, ratio_first=ratio_first, ratio_next=ratio_next, axis=axis)
    del(c)
    factors_triple = (None, None, None)
    # Calculate computational time and error of approximations:
    for i, chunk in enumerate(splitted_c):
        target_matrix = chunk if i == 0 else np.concatenate([target_matrix, chunk], axis=axis)
        factors_triple, comp_time = triple_func(
            i, axis, *factors_triple, target_matrix, chunk, rank, triple_func_params
        )

        time_dynamics.append(comp_time)
        err_dynamics.append(sqrt_err_relative_mtx(svd2mtx(*factors_triple), target_matrix))
        shape_dynamics.append(target_matrix.shape)

    return _get_experimental_results(time_dynamics, err_dynamics, size, rank, shape_dynamics)

######### About adding new rows/columns: #########
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

def get_incremental_svd_triple_block(step, axis, u, s, vt, target_matrix, chunk, rank, params):
    if step == 0:
        factors_triple, comp_time = elapsed_time(random_svd, chunk, rank, **params)
    else:
        factors_triple = u, s, vt
        factors_triple, comp_time = elapsed_time(update_svd_new_submatrix, *factors_triple, chunk)
    return factors_triple, comp_time

def get_dynamics_mtx(size, rank, ratio_first, ratio_next, triple_func, triple_func_params):
    time_dynamics = []
    err_dynamics = []
    shape_dynamics = []
    # Construct target dense matrix and split into chunks:
    small_rank = max(int(rank // (1 / ratio_next + 1)), 1)
    c = generate_matrix((size, size), small_rank)
    splitted_c = split_mtx_block_diag(c, ratio_first, ratio_next)
    del(c)
    factors_triple = (None, None, None)
    # Calculate computational time and error of approximations:
    for i, chunk in enumerate(splitted_c):
        target_matrix = chunk if i == 0 else block_diag(*splitted_c[:i+1])
        factors_triple, comp_time = triple_func(
            i, None, *factors_triple, target_matrix, chunk, rank, triple_func_params
        )

        time_dynamics.append(comp_time)
        err_dynamics.append(sqrt_err_relative_mtx(svd2mtx(*factors_triple), target_matrix))
        shape_dynamics.append(target_matrix.shape)

    return _get_experimental_results(time_dynamics, err_dynamics, size, rank, shape_dynamics)

######### About PSI step: #########
def svd_step(
    a: Union[csr_matrix, np.ndarray],
    rank: int ,
    n_power_iter: int,
    oversampling_factor: int,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(a, csr_matrix):
        u, s, vt = svds(a, k=rank)
    elif isinstance(a, np.ndarray):
        u, s, vt = random_svd(a, rank, n_power_iter, oversampling_factor, seed=seed)
    else:
        raise ValueError("Bad type of argument 'a'")
    return u, s, vt

def get_svd_dynamics(
    dynamics_list: List[np.ndarray],
    rank: int = 20
) -> Dict[str, List[float]]:
    time_dynamics = []
    err_dynamics = []
    timestamps = []
    for i, mtx in enumerate(dynamics_list):
        factors_triple, comp_time = elapsed_time(svd_step, mtx, rank=rank, **SVD_PARAMS)
        time_dynamics.append(comp_time)
        err_dynamics.append(sqrt_err_relative_mtx(svd2mtx(*factors_triple), mtx))
        timestamps.append(i)
    return _get_experimental_results(time_dynamics, err_dynamics, mtx.shape[0], rank, timestamps)

def get_psi_dynamics(
    dynamics_list: List[np.ndarray],
    rank: int = 20
) -> Dict[str, List[float]]:
    time_dynamics = []
    err_dynamics = []
    timestamps = []
    for i, a_i in enumerate(dynamics_list):
        if i == 0:
            a_i0 = a_i
            factors_triple, comp_time = elapsed_time(svd_step, a_i0, rank=rank, **SVD_PARAMS)
            time_dynamics.append(comp_time)
            err_dynamics.append(sqrt_err_relative_mtx(svd2mtx(*factors_triple), a_i0))
            timestamps.append(i)
            factors_triple = (factors_triple[0], np.diag(factors_triple[1]), factors_triple[2])
        else:
            da = a_i - a_i0
            factors_triple, comp_time = elapsed_time(psi_step, *factors_triple, da)
            time_dynamics.append(comp_time)
            u, s, vt = factors_triple
            err_dynamics.append(sqrt_err_relative_mtx(u.dot(s.dot(vt)), a_i))
            timestamps.append(i)
            a_i0 = a_i
    return _get_experimental_results(time_dynamics, err_dynamics, a_i0.shape[0], rank, timestamps)

class NumericalExperiment:
    def __init__(
        self,
        t0: float = 0.0,
        step_size: float = 0.2,
        size: int = 100,
        block_size: int = 10,
        eps: float = 1e-3,
        eps_block: float = 0.5,
        seed: int = 1
    ):
        self.t0 = t0
        self.ss = step_size
        self.size = size

        self.a1 = self._get_const_mtx((size,)*2, (block_size,)*2, eps, eps_block, seed + 1)
        self.a2 = self._get_const_mtx((size,)*2, (block_size,)*2, eps, eps_block, seed + 2)

        self.t1 = self._get_skew_symmetric((size,)*2, 1, seed + 1)
        self.t2 = self._get_skew_symmetric((size,)*2, 1, seed + 2)

        self.dynamics_list = None
        self.n_steps = None

    def _get_const_mtx(
        self,
        shape: Tuple[int, int] = (100, 100),
        shape_block: Tuple[int, int] = (10, 10),
        eps: float = 1e-3,
        eps_block: float = 0.5,
        seed: int = 13
    ) -> np.ndarray:
        random_state = np.random.RandomState(seed)
        a0 = eps_block * random_state.rand(*shape_block)
        a0[np.diag_indices_from(a0)] += 1

        a1 = eps * random_state.rand(*shape)
        a1r, a1c = shape_block
        a1[:a1r, :a1c] += a0
        return a1

    def _get_skew_symmetric(
        self,
        shape: Tuple[int, int] = (100, 100),
        scale: float = 1,
        seed: int = 13
    )-> np.ndarray:
        random_state = np.random.RandomState(seed)
        a = scale * random_state.rand(*shape)
        return 0.5*(a - a.T)

    def _step(self, t, q1, q2) -> np.ndarray:
        # get dynamic matrix in the current state:
        a3 = self.a1 + np.exp(t)*self.a2
        a = q1.dot(a3).dot(q2.T)
        return a

    def _solve_ivp(
        self,
        q_n:  np.ndarray,
        step_size: float,
        mix_mtx: np.ndarray, 
    ) -> np.ndarray:
        """ Solve: dQ/dt = TQ -> get Q_{n+1} = (I + hT)Q_n"""
        ht = step_size * mix_mtx
        ht[np.diag_indices_from(ht)] += 1
        return ht.dot(q_n)

    def get_dynamics_list(self, n_steps: int = 6) -> List[np.ndarray]:
        q1 = np.identity(self.size)
        q2 = np.identity(self.size)
        t = self.t0
        dynamics_list = []
        for _ in range(n_steps):
            dynamics_list.append(self._step(t, q1, q2))
            t += self.ss
            q1 = self._solve_ivp(q1, self.ss, self.t1)
            q2 = self._solve_ivp(q2, self.ss, self.t2)
        return dynamics_list

    def compare_approximations(self, n_steps: int = 6, rank: int = 10, plot_bool: bool = False):
        dynamics_list = self.get_dynamics_list(n_steps)
        dict_svd = get_svd_dynamics(dynamics_list, rank)
        dict_psi = get_psi_dynamics(dynamics_list, rank)
        # Get x and y for plotting:
        x = np.linspace(self.t0, self.t0 + (n_steps - 1)*self.ss, n_steps)
        y1, y2 = dict_svd, dict_psi
        if plot_bool:
            # Computational Time:
            xyl_list = [
                {'x': x, 'y': y1['ctime_dynamics'], 'label': 'SVD'},
                {'x': x, 'y': y2['ctime_dynamics'], 'label': 'PSI'},
            ]
            get_plot(
                xyl_list,
                rc_params=PARAMS, 
                xlabel='Timestamp', 
                ylabel='Computational Time (sec)', 
                title=f'Computational Time Dynamics'
            )
            # Relative error:
            xyl_list = [
                {'x': x, 'y': y1['error_dynamics'], 'label': 'SVD'},
                {'x': x, 'y': y2['error_dynamics'], 'label': 'PSI'},
            ]
            get_plot(
                xyl_list,
                rc_params=PARAMS, 
                xlabel='Timestamp', 
                ylabel='Relative Error', 
                title=f'Relative Error Dynamics'
            )
            print(
                "Overall computational time:\n"
                + f"SVD: {np.sum(y1['ctime_dynamics'])}\n"
                f"PSI: {np.sum(y2['ctime_dynamics'])}\n"
            )
        return dict_svd, dict_psi

######### About plots: #########
def get_plot(xyl_list, rc_params, xlabel, ylabel, title, marker='*'):
    with mpl.rc_context(rc_params):
        _, ax = get_fig_ax()
        for entity in xyl_list:
            add_plot_ax(ax, entity['x'], entity['y'], label=entity['label'], marker=marker)
        set_ax(ax, xlabel, ylabel, title) 
    
def matrix_plot_2d(axs, fig, matrix, title):
    cax = axs.matshow(matrix)
    fig.colorbar(cax, ax=axs)
    axs.set_xlabel('cols', fontsize=16)
    axs.set_ylabel('rows', fontsize=16)
    axs.set_title(title, fontsize=18)

def plot_dynamics_2d(dynamics_list: List[np.ndarray]):
    n_rows = len(dynamics_list) // 3
    n_cols = 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 11))
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i*n_cols + j
            matrix_plot_2d(axs[i, j], fig, dynamics_list[idx], f"A(t{idx})")

def matrix_plot_3d(fig, matrix, pos, title, **kwargs):
    (x, y) = np.meshgrid(
        np.arange(matrix.shape[0]),
        np.arange(matrix.shape[1])
    )
    ax = fig.add_subplot(*pos, projection='3d')
    ax.plot_surface(x, y, matrix, **kwargs)
    ax.set_xlabel('X (cols)', fontsize=14)
    ax.set_ylabel('Y (rows)', fontsize=14)
    ax.set_zlabel('Z (values)', fontsize=14)
    ax.set_title(title, fontsize=18)

def plot_dynamics_3d(dynamics_list: List[np.ndarray]):
    n_rows = len(dynamics_list) // 3
    n_cols = 3
    fig = plt.figure(figsize=(20, 10))
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i*n_cols + j
            matrix_plot_3d(fig, dynamics_list[idx], (2, 3, idx+1), f"A(t{idx})")
