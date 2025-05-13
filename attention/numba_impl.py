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
def matmul_blocked(A, B, block_size):
    n, m = A.shape
    m2, p = B.shape
    assert m == m2
    C = np.zeros((n, p), dtype=A.dtype)
    
    for i0 in range(0, n, block_size):
        for j0 in range(0, p, block_size):
            for k0 in range(0, m, block_size):
                i_max = min(i0 + block_size, n)
                j_max = min(j0 + block_size, p)
                k_max = min(k0 + block_size, m)
                for i in range(i0, i_max):
                    for k in range(k0, k_max):
                        for j in range(j0, j_max):
                            C[i, j] += A[i, k] * B[k, j]
    return C

@njit
def attention(Q, K, V, block_size=32):
    dtype = Q.dtype
    Q = Q.astype(dtype)
    K = K.astype(dtype)
    V = V.astype(dtype)
    d = dtype.type(Q.shape[1])  # conversion int -> float (même type que Q)
    
    scores = matmul_blocked(Q, K.T, block_size) / np.sqrt(d)
    weights = softmax_2d(scores)
    return matmul_blocked(weights, V, block_size)