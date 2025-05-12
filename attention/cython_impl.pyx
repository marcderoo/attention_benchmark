# cython: boundscheck=False, wraparound=False, cdivision=True
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport expf, sqrtf, exp, sqrt

# ----------------------
# Softmax row helpers
# ----------------------
cdef inline void softmax_row_f32(float* row, int n) nogil noexcept:
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

cdef inline void softmax_row_f64(double* row, int n) nogil noexcept:
    cdef int i
    cdef double max_val = row[0]
    for i in range(1, n):
        if row[i] > max_val:
            max_val = row[i]

    cdef double sum_exp = 0.0
    for i in range(n):
        row[i] = exp(row[i] - max_val)
        sum_exp += row[i]

    for i in range(n):
        row[i] /= sum_exp

# ----------------------
# Attention core function for float32
# ----------------------
cdef np.ndarray[np.float32_t, ndim=2] attention_f32(np.ndarray[np.float32_t, ndim=2] Q,
                                                    np.ndarray[np.float32_t, ndim=2] K,
                                                    np.ndarray[np.float32_t, ndim=2] V,
                                                    int num_threads=0):
    cdef int n = Q.shape[0]
    cdef int d = Q.shape[1]
    cdef int dv = V.shape[1]
    cdef float scale = 1.0 / sqrtf(d)

    cdef np.ndarray[np.float32_t, ndim=2] scores = np.empty((n, n), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] output = np.zeros((n, dv), dtype=np.float32)

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

    for i in range(n):
        softmax_row_f32(&scores[i, 0], n)

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

# ----------------------
# Attention core function for float64
# ----------------------
cdef np.ndarray[np.float64_t, ndim=2] attention_f64(np.ndarray[np.float64_t, ndim=2] Q,
                                                    np.ndarray[np.float64_t, ndim=2] K,
                                                    np.ndarray[np.float64_t, ndim=2] V,
                                                    int num_threads=0):
    cdef int n = Q.shape[0]
    cdef int d = Q.shape[1]
    cdef int dv = V.shape[1]
    cdef double scale = 1.0 / sqrt(d)

    cdef np.ndarray[np.float64_t, ndim=2] scores = np.empty((n, n), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] output = np.zeros((n, dv), dtype=np.float64)

    cdef int i, j, k
    cdef double* q_ptr
    cdef double* k_ptr
    cdef double* s_ptr
    cdef double dotp

    for i in range(n):
        q_ptr = &Q[i, 0]
        s_ptr = &scores[i, 0]
        for j in range(n):
            k_ptr = &K[j, 0]
            dotp = 0.0
            for k in range(d):
                dotp += q_ptr[k] * k_ptr[k]
            s_ptr[j] = dotp * scale

    for i in range(n):
        softmax_row_f64(&scores[i, 0], n)

    cdef double* w_ptr
    cdef double* v_ptr
    cdef double* o_ptr

    for i in range(n):
        w_ptr = &scores[i, 0]
        o_ptr = &output[i, 0]
        for j in range(n):
            v_ptr = &V[j, 0]
            dotp = w_ptr[j]
            for k in range(dv):
                o_ptr[k] += dotp * v_ptr[k]

    return output

# ----------------------
# Dispatcher
# ----------------------
def attention(Q, K, V, int num_threads=0):
    if Q.dtype == np.float32:
        return attention_f32(Q, K, V, num_threads)
    elif Q.dtype == np.float64:
        return attention_f64(Q, K, V, num_threads)
    else:
        raise TypeError("Only float32 and float64 dtypes are supported")