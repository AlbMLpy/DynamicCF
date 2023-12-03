import numpy as np
from numba import jit

NO_PYTHON = True

@jit(nopython=NO_PYTHON)
def mrr(rec_array, test_items):  
    total = 0.0  
    for i in range(len(test_items)):   
        test_item = test_items[i]  
        u_recs = rec_array[i]  
        for j, ui_rec in enumerate(u_recs):  
            if ui_rec == test_item:  
                total += 1 / (1 + j) 
                break 
    return total / len(test_items)  
 
@jit(nopython=NO_PYTHON)
def hr(rec_array, test_items):  
    total = 0.0  
    for i in range(len(test_items)):   
        test_item = test_items[i]  
        u_recs = rec_array[i]  
        if test_item in u_recs:  
            total += 1  
    return total / len(test_items)  

@jit(nopython=NO_PYTHON)
def wji(u, v):
    numer, denom = 0.0, 0.0
    for i in range(len(u)):
        numer += min(u[i], v[i])
        denom += max(u[i], v[i])
    return numer / denom

@jit(nopython=NO_PYTHON)
def wji_sim(arr1, arr2, N):
    total = 0.0 
    n_users, topn = arr1.shape
    for i in range(n_users):   
        bow1, bow2 = np.zeros(N), np.zeros(N)
        bow1[arr1[i]] = 1 / np.arange(1, topn + 1)
        bow2[arr2[i]] = 1 / np.arange(1, topn + 1)
        total += wji(bow1, bow2)   
    return total / n_users 
