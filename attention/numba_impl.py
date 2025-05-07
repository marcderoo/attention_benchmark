import numpy as np
from numba import njit, prange

@njit
def softmax_row(x):
    max_val = np.max(x)
    exps = np.exp(x - max_val)
    return exps / np.sum(exps)

@njit(parallel=True)
def attention(Q, K, V):
    n, d = Q.shape
    _, dk = V.shape
    out = np.zeros((n, dk), dtype=np.float32)

    for i in prange(n):
        score = np.zeros(n, dtype=np.float32)
        for j in range(n):
            for k in range(d):
                score[j] += Q[i, k] * K[j, k]
        for j in range(n):
            score[j] /= np.sqrt(d)

        weights = softmax_row(score)

        for j in range(n):
            for k in range(dk):
                out[i, k] += weights[j] * V[j, k]

    return out
