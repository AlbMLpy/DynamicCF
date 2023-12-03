from time import time

import numpy as np

def elapsed_time(func, *args, **kwargs):
    s_time = time()
    result = func(*args, **kwargs)
    e_time = time()
    return result, e_time - s_time

def sliding_window(arr: np.array, win_len: int, mode: str = 'sum') -> np.array:
    len_arr = len(arr)
    if len_arr < win_len:
        return arr
    res = []
    if mode == 'sum':
        _f = np.sum
    elif mode == 'mean':
        _f = np.mean
    else:
        raise NotImplementedError()
    for i in range(len_arr - win_len + 1):
        res.append(_f(arr[i:i + win_len]))
    return np.array(res)

def rel_norm(a, b, ord_norm: str = 'fro'):
    return np.linalg.norm(
        a-b, 
        ord=ord_norm) / np.linalg.norm(a, ord=ord_norm
    )
