# cython_impl.pyx
# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt
from cython.parallel import prange
from cython cimport nogil, gil

cdef np.ndarray[np.float32_t, ndim=1] softmax_row(np.ndarray[np.float32_t, ndim=1] x):
    cdef int n = x.shape[0]
    cdef np.ndarray[np.float32_t, ndim=1] exps = np.empty(n, dtype=np.float32)
    cdef float max_val = x[0]
    cdef float sum_exps = 0.0
    cdef int i

    for i in range(1, n):
        if x[i] > max_val:
            max_val = x[i]

    for i in range(n):
        exps[i] = exp(x[i] - max_val)
        sum_exps += exps[i]

    for i in range(n):
        exps[i] /= sum_exps

    return exps

def attention(np.ndarray[np.float32_t, ndim=2] Q,
              np.ndarray[np.float32_t, ndim=2] K,
              np.ndarray[np.float32_t, ndim=2] V):

    cdef int n = Q.shape[0]
    cdef int d = Q.shape[1]
    cdef int dk = V.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2] out = np.zeros((n, dk), dtype=np.float32)
    cdef float scale = 1.0 / sqrt(d)

    cdef int i, j, k
    cdef np.ndarray[np.float32_t, ndim=1] score
    cdef np.ndarray[np.float32_t, ndim=1] weights

    for i in prange(n, nogil=True):
        with gil:
            score = np.zeros(n, dtype=np.float32)

        for j in range(n):
            for k in range(d):
                score[j] += Q[i, k] * K[j, k]
            score[j] *= scale

        with gil:
            weights = softmax_row(score)

        for j in range(n):
            for k in range(dk):
                out[i, k] += weights[j] * V[j, k]

    return out