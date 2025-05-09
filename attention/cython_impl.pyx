# cython: boundscheck=False, wraparound=False, cdivision=True
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport expf, sqrtf

# ------------------------------------------------------------------------------
# Compute softmax of a single row (length n) in-place: row -> softmax(row)
# ------------------------------------------------------------------------------
cdef inline void softmax_row(float* row, int n) nogil noexcept:
    cdef int i
    cdef float max_val = row[0]
    for i in range(1, n):
        if row[i] > max_val:
            max_val = row[i]

    cdef float sum_exp = 0.0
    for i in range(n):
        row[i] = expf(row[i] - max_val)
        sum_exp += row[i]

    for i in range(n):
        row[i] /= sum_exp

# ------------------------------------------------------------------------------
# Main attention function
# ------------------------------------------------------------------------------
def attention(np.ndarray[np.float32_t, ndim=2] Q, 
              np.ndarray[np.float32_t, ndim=2] K,
              np.ndarray[np.float32_t, ndim=2] V,
              int num_threads=0) -> np.ndarray:
    """
    Scaled dot-product attention in pure C with OpenMP optional.
    Q: (n, d)
    K: (n, d)
    V: (n, dv)
    """
    cdef int n = Q.shape[0]
    cdef int d = Q.shape[1]
    cdef int dv = V.shape[1]
    cdef float scale = 1.0 / sqrtf(d)

    # Allocate output arrays
    cdef np.ndarray[np.float32_t, ndim=2] scores = np.empty((n, n), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] output = np.zeros((n, dv), dtype=np.float32)

    # Compute scaled Q @ K^T
    cdef int i, j, k
    cdef float* q_ptr
    cdef float* k_ptr
    cdef float* s_ptr
    cdef float dotp

    for i in range(n):
        q_ptr = &Q[i, 0]
        s_ptr = &scores[i, 0]
        for j in range(n):
            k_ptr = &K[j, 0]
            dotp = 0.0
            for k in range(d):
                dotp += q_ptr[k] * k_ptr[k]
            s_ptr[j] = dotp * scale

    # Apply softmax to each row
    for i in range(n):
        softmax_row(&scores[i, 0], n)

    # Compute output = scores @ V
    cdef float* w_ptr
    cdef float* v_ptr
    cdef float* o_ptr

    for i in range(n):
        w_ptr = &scores[i, 0]
        o_ptr = &output[i, 0]
        for j in range(n):
            v_ptr = &V[j, 0]
            dotp = w_ptr[j]
            for k in range(dv):
                o_ptr[k] += dotp * v_ptr[k]

    return output