# cython_impl.pyx
# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt
from cython.parallel import prange
from cython cimport nogil, gil

# Fonction softmax sur une ligne
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

# Fonction d'attention
def attention(np.ndarray[np.float32_t, ndim=2] Q,
              np.ndarray[np.float32_t, ndim=2] K,
              np.ndarray[np.float32_t, ndim=2] V):

    cdef int n = Q.shape[0]
    cdef int d = Q.shape[1]
    cdef int dk = V.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2] out = np.zeros((n, dk), dtype=np.float32)
    cdef float scale = 1.0 / sqrt(d)

    # Pointeurs pour accès rapide
    cdef float* Q_ptr = <float*> Q.data
    cdef float* K_ptr = <float*> K.data
    cdef float* V_ptr = <float*> V.data
    cdef float* out_ptr = <float*> out.data

    # Déclarations nécessaires en dehors de prange
    cdef int i, j, k
    cdef np.ndarray[np.float32_t, ndim=1] score
    cdef np.ndarray[np.float32_t, ndim=1] weights
    cdef float* score_ptr
    cdef float* weights_ptr

    for i in prange(n, nogil=True):
        with gil:
            score = np.zeros(n, dtype=np.float32)
            score_ptr = <float*> score.data

        for j in range(n):
            for k in range(d):
                score_ptr[j] += Q_ptr[i * d + k] * K_ptr[j * d + k]
            score_ptr[j] *= scale

        with gil:
            weights = softmax_row(score)
            weights_ptr = <float*> weights.data

        for j in range(n):
            for k in range(dk):
                out_ptr[i * dk + k] += weights_ptr[j] * V_ptr[j * dk + k]

    return out