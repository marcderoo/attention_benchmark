# cython_impl.pyx
import numpy as np
cimport numpy as np
from libc.math cimport expf, sqrtf

# Activer le support OpenMP
cdef extern from "omp.h":
    int omp_get_max_threads()
    int omp_get_num_threads()
    int omp_get_thread_num()
    void omp_set_num_threads(int)

# Déclaration de l'interface principale
def attention(np.ndarray[np.float32_t, ndim=2, mode="c"] Q,
              np.ndarray[np.float32_t, ndim=2, mode="c"] K,
              np.ndarray[np.float32_t, ndim=2, mode="c"] V):
    """
    Compute scaled dot-product attention using Cython with OpenMP.
    Q, K, V: shape (n, d) and (n, dk)
    """
    cdef int n = Q.shape[0]
    cdef int d = Q.shape[1]
    cdef int dk = V.shape[1]
    cdef float scale = 1.0 / sqrtf(d)

    cdef np.ndarray[np.float32_t, ndim=2] out = np.zeros((n, dk), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] score = np.zeros(n, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] weights = np.zeros(n, dtype=np.float32)

    cdef int i, j, k
    cdef float max_val, sum_exp, s

    # Attention avec OpenMP (parallélisation sur i)
    for i in range(n):
        # Calcul du score[i, j] = dot(Q[i], K[j])
        for j in range(n):
            score[j] = 0
            for k in range(d):
                score[j] += Q[i, k] * K[j, k]
            score[j] *= scale

        # Softmax
        max_val = score[0]
        for j in range(1, n):
            if score[j] > max_val:
                max_val = score[j]

        sum_exp = 0.0
        for j in range(n):
            weights[j] = expf(score[j] - max_val)
            sum_exp += weights[j]
        for j in range(n):
            weights[j] /= sum_exp

        # Pondération par V
        for j in range(n):
            for k in range(dk):
                out[i, k] += weights[j] * V[j, k]

    return out
