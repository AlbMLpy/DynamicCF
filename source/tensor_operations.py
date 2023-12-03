from typing import Tuple

import numpy as np
from numba import jit, njit, prange
from scipy.sparse import coo_matrix
from scipy.linalg import khatri_rao

D_TYPE = np.float64
EPS = 1e-14
NO_PYTHON=True

def unfold_sparse(
    inds: np.ndarray,
    vals: np.ndarray,
    shape: Tuple[int, int, int],
    mode: int,
) -> coo_matrix:
    """
    TODO
    """
    n_rows, n_cols = inds.shape
    vals_size = vals.size
    if n_rows != vals_size:
        raise RuntimeError(f"inds and vals are incompatible in size: inds: {n_rows} != vals: {vals_size}")
    if (mode >= n_cols) or mode < 0:
        raise RuntimeError(f"bad mode: {mode}; must be >= 0 and < {n_cols}")
    if n_cols > 3:
        raise NotImplementedError("Considering only 3-way tensors")

    modes = np.arange(3)
    mode_a, mode_b = modes[modes != mode]

    rows = inds[:, mode]
    cols = inds[:, mode_a] + shape[mode_a]*inds[:, mode_b]
    return coo_matrix((vals, (rows, cols)), shape=(shape[mode], shape[mode_a]*shape[mode_b]), dtype=D_TYPE)

def unfold_dense(
    tensor: np.ndarray,
    mode: int,
) -> np.ndarray:
    """
    TODO
    """
    modes = np.arange(3)
    mode_a, mode_b = modes[modes != mode]
    shape = tensor.shape
    return tensor.transpose(mode, mode_b, mode_a).reshape(shape[mode], shape[mode_a]*shape[mode_b])

def fold_dense(
    mtx: np.ndarray,
    shape,
    mode: int
) -> np.ndarray:
    """
    TODO
    """
    modes = np.arange(3)
    mode_a, mode_b = modes[modes != mode]
    axes = (mode, mode_b, mode_a) if mode != 1 else (mode_b, mode_a, mode)
    return mtx.reshape(shape[mode], shape[mode_b], shape[mode_a]).transpose(axes)

@jit(nopython=NO_PYTHON)
def construct_core(
    inds: np.ndarray,
    vals: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
    u3: np.ndarray,
) -> np.ndarray:
    """
    TODO
    """
    _, r1 = u1.shape
    _, r2 = u2.shape
    _, r3 = u3.shape

    core = np.zeros(shape=(r1, r2, r3), dtype=D_TYPE)

    for ind in range(len(vals)):
        i1 = inds[ind, 0]
        i2 = inds[ind, 1]
        i3 = inds[ind, 2]
        vi = vals[ind]

        u1_i1 = u1[i1]
        u2_i2 = u2[i2]
        u3_i3 = u3[i3]

        for i in range(r1):
            vi_u1_i1i = vi * u1_i1[i]
            for j in range(r2):
                vi_u1_i1i_u2_i2j = vi_u1_i1i * u2_i2[j]
                for k in range(r3):
                    core[i, j, k] += vi_u1_i1i_u2_i2j *  u3_i3[k]
    return core

@njit(parallel=True)
def construct_core_parallel(
    inds: np.ndarray,
    vals: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
    u3: np.ndarray,
) -> np.ndarray:
    """
    TODO
    """
    _, r1 = u1.shape
    _, r2 = u2.shape
    _, r3 = u3.shape
   
    core = np.zeros(shape=(r1, r2, r3), dtype=np.float64)

    for i in prange(r1):
        for ind in range(len(vals)):
            i1 = inds[ind, 0]
            i2 = inds[ind, 1]
            i3 = inds[ind, 2]
            vi = vals[ind]

            u1_i1 = u1[i1]
            u2_i2 = u2[i2]
            u3_i3 = u3[i3]

            vi_u1_i1i = vi * u1_i1[i]
            for j in range(r2):
                vi_u1_i1i_u2_i2j = vi_u1_i1i * u2_i2[j]
                for k in range(r3):
                    core[i, j, k] += vi_u1_i1i_u2_i2j *  u3_i3[k]
    return core

@njit(parallel=True)
def construct_core_parallel_dense(
    x: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
    u3: np.ndarray,
) -> np.ndarray:
    """
    TODO
    """
    _, r1 = u1.shape
    _, r2 = u2.shape
    _, r3 = u3.shape
    n1, n2, n3 = x.shape
   
    core = np.zeros(shape=(r1, r2, r3), dtype=np.float64)

    for i in prange(r1):
        for i1 in range(n1):
            u1_i1 = u1[i1]
            for i2 in range(n2):
                u2_i2 = u2[i2]
                for i3 in range(n3):
                    vi = x[i1, i2, i3]
                    u3_i3 = u3[i3]

                    vi_u1_i1i = vi * u1_i1[i]
                    for j in range(r2):
                        vi_u1_i1i_u2_i2j = vi_u1_i1i * u2_i2[j]
                        for k in range(r3):
                            core[i, j, k] += vi_u1_i1i_u2_i2j *  u3_i3[k]
    return core

@jit(nopython=NO_PYTHON)
def _double_tensordot(
    inds: np.ndarray,
    vals: np.ndarray,
    u: np.ndarray,
    v: np.ndarray, 
    mode0: int, 
    mode1: int, 
    mode2: int, 
    res: np.ndarray,
):
    new_shape1 = u.shape[1]
    new_shape2 = v.shape[1]
    for i in range(len(vals)):
        i0 = inds[i, mode0]
        i1 = inds[i, mode1]
        i2 = inds[i, mode2]
        vi = vals[i]

        u_i1 = u[i1]
        v_i2 = v[i2]
        for j in range(new_shape1):
            vi_u_i1j = vi * u_i1[j]
            for k in range(new_shape2):
                res[i0, j, k] += vi_u_i1j * v_i2[k]

def tensordot2(
    inds: np.ndarray,
    vals: np.ndarray,
    shape: Tuple[int, int, int],
    u: np.ndarray,
    v: np.ndarray,
    modes: Tuple[int, int],
) -> np.ndarray:
    """
    TODO
    """
    if len(shape) > 3:
        raise NotImplementedError("Considering only 3-way tensors")
    
    mode1, mode2 = modes
    mode0 = 3 - sum(modes)
    if mode1 not in (0, 1, 2):
        raise RuntimeError(f"Bad modes[0]: {mode1}; Should be in (0, 1, 2)")
    if mode2 not in (0, 1, 2):
        raise RuntimeError(f"Bad modes[1]: {mode2}; Should be in (0, 1, 2)")
    if mode1 == mode2:
        raise RuntimeError("Modes must be different")

    _, u_cols = u.shape
    _, v_cols = v.shape
    res = np.zeros(shape=(shape[mode0], u_cols, v_cols), dtype=D_TYPE)

    _double_tensordot(inds, vals, u, v, mode0, mode1, mode2, res)
    return res

@jit(nopython=True)
def tensordot(
    inds: np.ndarray,
    vals: np.ndarray,
    shape: Tuple[int, int, int],
    u: np.ndarray, 
    mode: int,
):
    n_rows, _ = u.shape
    if mode == 0:
        new_shape = (n_rows, shape[1], shape[2])
    elif mode == 1:
        new_shape = (shape[0], n_rows, shape[2])
    elif mode == 2:
        new_shape = (shape[0], shape[1], n_rows)

    result = np.zeros(shape=new_shape, dtype=D_TYPE)
    for j in range(n_rows):
        u_j = u[j]
        for i in range(len(vals)):
            i0, i1, i2 = inds[i]
            vi = vals[i]
            if mode == 0:
                u_ji = u_j[i0]
                i0 = j
            elif mode == 1:
                u_ji = u_j[i1]
                i1 = j
            elif mode == 2:
                u_ji = u_j[i2]
                i2 = j
            result[i0, i1, i2] += vi * u_ji
    return result

def sparse_tensor_double_tensor(
    inds: np.ndarray,
    vals: np.ndarray, 
    shape, 
    q: np.ndarray, 
    mode: int, 
    u1: np.ndarray, 
    u2: np.ndarray
):
    modes = np.arange(3)
    mode_a, mode_b = modes[modes != mode]
    result = np.zeros(shape=(shape[mode], q.shape[1]), dtype=D_TYPE)
    q_t = q.T

    n = len(inds)
    n_points_per_step = min(10000, n)
    n_steps = n // n_points_per_step

    for step in range(n_steps):
        st = step*n_points_per_step
        chunk = inds[st: st + n_points_per_step]
        i1, i2 = chunk[:, mode_a], chunk[:, mode_b]
        vec_all = khatri_rao(u2[i2].T, u1[i1].T)
        
        vec_all = q_t.dot(vec_all)

        for li, i in enumerate(range(st, st + n_points_per_step)):
            coords, val = inds[i], vals[i]
            i1, i2, r_ind = coords[mode_a], coords[mode_b], coords[mode]
            if val != 1.0:
                result[r_ind] += val * vec_all[:, li]
            else:
                result[r_ind] += vec_all[:, li]
    return result

@jit(nopython=NO_PYTHON)
def kron_vec_jit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    la, lb = len(a), len(b)
    vec = np.empty(la*lb, dtype=D_TYPE)
    for i in range(la):
        a_i = a[i]
        lb_i = lb * i
        for k in range(lb):
            vec[k + lb_i] = a_i * b[k]
    return vec
   
@jit(nopython=NO_PYTHON)
def sparse_tensor_double_tensor_old(
    inds: np.ndarray,
    vals: np.ndarray, 
    shape: Tuple[int, int, int], 
    q: np.ndarray, 
    mode: int, 
    u1: np.ndarray, 
    u2: np.ndarray
):
    modes = np.arange(3)
    mode_a, mode_b = modes[modes != mode]

    n_len = len(inds)

    n_rows, n_cols = shape[mode], q.shape[1]
    result = np.zeros(shape=(n_rows, n_cols), dtype=D_TYPE)
    q_t = q.T
    cache = {}
    for i in range(n_len):
        coords, val = inds[i], vals[i]
        r_ind = coords[mode]
        i1, i2 = coords[mode_a], coords[mode_b]
        if (i2, i1) not in cache:
            v_j = kron_vec_jit(u2[i2], u1[i1])
            vh_j = q_t.dot(v_j)
            cache[(i2, i1)] = vh_j
        result[r_ind] += val * cache[(i2, i1)]
    return result


@jit(nopython=NO_PYTHON)
def get_tucker_element(
    i: int,
    j: int,
    k: int,
    u1: np.ndarray,
    u2: np.ndarray,
    u3: np.ndarray,
    core: np.ndarray
) -> float:
    r1, r2, r3 = u1.shape[1], u2.shape[1], u3.shape[1]
    res = 0.0

    u1_i = u1[i]
    u2_j = u2[j]
    u3_k = u3[k]
    for i1 in range(r1):
        u1_i_i1 = u1_i[i1]
        for i2 in range(r2):
            u1_i_i1_u2_j_i2 = u1_i_i1 * u2_j[i2]
            for i3 in range(r3):
                res += core[i1, i2, i3] * u1_i_i1_u2_j_i2 * u3_k[i3] 
    return res

@jit(nopython=NO_PYTHON)
def construct_tensor_from_tucker(
    u1: np.ndarray,
    u2: np.ndarray,
    u3: np.ndarray,
    core: np.ndarray
) -> np.ndarray:
    n1, n2, n3 = u1.shape[0], u2.shape[0], u3.shape[0]
    result = np.empty(shape=(n1, n2, n3))
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                result[i, j, k] = get_tucker_element(i, j, k, u1, u2, u3, core)
    return result

@jit(nopython=NO_PYTHON)
def dense_to_sparse_tensor(tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n1, n2, n3 = tensor.shape
    inds = []
    values = []
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                values.append(tensor[i, j, k])
                inds.append([i, j, k])
    return np.array(inds), np.array(values)

@jit(nopython=NO_PYTHON)
def sparse_to_dense_tensor(inds, vals, shape):
    res = np.zeros(shape)
    for i in range(len(vals)):
        i1, i2, i3 = inds[i]
        res[i1, i2, i3] = vals[i]
    return res

@jit(nopython=NO_PYTHON)
def gen_hilbert_tensor(shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    inds = []
    values = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                inds.append((i, j, k))
                values.append(1 / (i + j + k + 1))
    inds = np.array(inds)
    values = np.array(values)
    return inds, values 

@jit(nopython=NO_PYTHON)
def _sqrt_err(
    inds: np.ndarray,
    vals: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
    u3: np.ndarray,
    core: np.ndarray
) -> float:
    result = 0.0
    for item in range(inds.shape[0]):
        i1, i2, i3 = inds[item]
        result += (vals[item] - get_tucker_element(i1, i2, i3, u1, u2, u3, core))**2        
    return np.sqrt(result)

@jit(nopython=NO_PYTHON)
def sqrt_err_relative(
    inds: np.ndarray,
    vals: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
    u3: np.ndarray,
    core: np.ndarray
) -> float:
    result = _sqrt_err(inds, vals, u1, u2, u3, core)         
    return result / np.sqrt((vals**2).sum() + EPS)
