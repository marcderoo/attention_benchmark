import numpy as np
from numba import njit

@njit
def softmax_2d(x):
    n, m = x.shape
    out = np.empty((n, m), dtype=x.dtype)
    for i in range(n):
        max_val = np.max(x[i])
        exps = np.exp(x[i] - max_val)
        out[i] = exps / np.sum(exps)
    return out

@njit
def attention(Q, K, V):
    dtype = Q.dtype
    Q = Q.astype(dtype)
    K = K.astype(dtype)
    V = V.astype(dtype)
    d = dtype.type(Q.shape[1])  
    scores = (Q @ K.T) / np.sqrt(d)
    weights = softmax_2d(scores)
    return weights @ V
