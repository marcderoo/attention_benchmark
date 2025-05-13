# cython: boundscheck=False, wraparound=False, cdivision=True
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport expf, sqrtf, exp, sqrt

# ------------------------------------------------------------------------------
# Compute softmax of a single row (length n) in-place: row -> softmax(row)
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# Matrix multiplication implementation
# ------------------------------------------------------------------------------
cdef extern from "mmat_impl.h":
    void mmat_impl_cpp(int n_row, int n_col, int k,
                       const float* p1, const float* p2, float* res,
                       int block_size, int version, int nb_threads)
    void mmat_impl_cpp(int n_row, int n_col, int k,
                       const double* p1, const double* p2, double* res,
                       int block_size, int version, int nb_threads)

cdef mmat_c_float(const float[:, ::1] a, const float [:, ::1] b, float [:, ::1] res,
                  int block_size, int version, int nb_threads):
    mmat_impl_cpp(a.shape[0], b.shape[1], a.shape[1], &a[0, 0], &b[0, 0], &res[0, 0],
                  block_size, version, nb_threads)

cdef mmat_c_double(const double[:, ::1] a, const double[:, ::1] b, double[:, ::1] res,
                   int block_size, int version, int nb_threads):    
    mmat_impl_cpp(a.shape[0], b.shape[1], a.shape[1], &a[0, 0], &b[0, 0], &res[0, 0],
                  block_size, version, nb_threads)

def mmat_impl(np.ndarray a, np.ndarray b, block_size=16, version=0, nb_threads=0):
    res = np.zeros((a.shape[0], b.shape[1]), dtype=a.dtype)
    if a.dtype == np.float32:
        mmat_c_float(a, b, res, block_size, version, nb_threads)
    elif a.dtype == np.float64:
        mmat_c_double(a, b, res, block_size, version, nb_threads)
    else:
        raise NotImplementedError(f"Not implemented for dtype={a.dtype}")
    return res

def mmat(a, b, block_size=16, version=0, nb_threads=0):
    """Matrix multiplication."""
    assert len(a.shape) == 2 == len(b.shape), (
        f"Only applies on matrices but a.shape={a.shape}, b.shape={b.shape}"
    )
    assert a.shape[1] == b.shape[0], (
        f"Shape mismatch: a.shape={a.shape}, b.shape={b.shape}"
    )
    assert a.dtype == b.dtype, f"Type mismatch a.dtype={a.dtype}, b.dtype={b.dtype}"
    assert a.flags["C_CONTIGUOUS"], "Matrix a must be contiguous"
    assert b.flags["C_CONTIGUOUS"], "Matrix b must be contiguous"
    return mmat_impl(a, b, block_size, version, nb_threads)

# ------------------------------------------------------------------------------
# Attention core functions that use mmat
# ------------------------------------------------------------------------------
cdef np.ndarray[np.float32_t, ndim=2] attention_f32(np.ndarray[np.float32_t, ndim=2] Q,
                                                   np.ndarray[np.float32_t, ndim=2] K,
                                                   np.ndarray[np.float32_t, ndim=2] V,
                                                   int nb_threads=0,
                                                   int block_size=16,
                                                   int version=1):
    cdef int n = Q.shape[0]
    cdef int d = Q.shape[1]
    cdef int dv = V.shape[1]
    cdef float scale = 1.0 / sqrtf(d)
    
    # Transposer K pour le produit Q @ K.T
    cdef np.ndarray[np.float32_t, ndim=2] KT = K.transpose().copy()
    
    # Calcul de scores = (Q @ K.T) * scale en utilisant mmat
    # Utiliser les paramètres block_size et version passés à la fonction
    cdef np.ndarray[np.float32_t, ndim=2] scores = mmat_impl(Q, KT, block_size, version, nb_threads)
    
    # Appliquer le scaling
    scores *= scale
    
    # Appliquer softmax sur chaque ligne
    for i in range(n):
        softmax_row_f32(&scores[i, 0], n)
    
    # Calcul de output = scores @ V en utilisant mmat
    # Utiliser les paramètres block_size et version passés à la fonction
    cdef np.ndarray[np.float32_t, ndim=2] output = mmat_impl(scores, V, block_size, version, nb_threads)
    
    return output

cdef np.ndarray[np.float64_t, ndim=2] attention_f64(np.ndarray[np.float64_t, ndim=2] Q,
                                                   np.ndarray[np.float64_t, ndim=2] K,
                                                   np.ndarray[np.float64_t, ndim=2] V,
                                                   int nb_threads=0,
                                                   int block_size=16,
                                                   int version=1):
    cdef int n = Q.shape[0]
    cdef int d = Q.shape[1]
    cdef int dv = V.shape[1]
    cdef double scale = 1.0 / sqrt(d)
    
    # Transposer K pour le produit Q @ K.T
    cdef np.ndarray[np.float64_t, ndim=2] KT = K.transpose().copy()
    
    # Calcul de scores = (Q @ K.T) * scale en utilisant mmat
    # Utiliser les paramètres block_size et version passés à la fonction
    cdef np.ndarray[np.float64_t, ndim=2] scores = mmat_impl(Q, KT, block_size, version, nb_threads)
    
    # Appliquer le scaling
    scores *= scale
    
    # Appliquer softmax sur chaque ligne
    for i in range(n):
        softmax_row_f64(&scores[i, 0], n)
    
    # Calcul de output = scores @ V en utilisant mmat
    # Utiliser les paramètres block_size et version passés à la fonction
    cdef np.ndarray[np.float64_t, ndim=2] output = mmat_impl(scores, V, block_size, version, nb_threads)
    
    return output

# ------------------------------------------------------------------------------
# Dispatcher for attention
# ------------------------------------------------------------------------------
def attention(Q, K, V, int nb_threads=0, int block_size=32, int version=1):
    """
    Scaled dot-product attention using optimized matrix multiplication.
    Q: (n, d)
    K: (n, d)
    V: (n, dv)
    
    block_size: Taille des blocs pour la multiplication matricielle
    version: Version de l'algorithme (0 = standard, 1 = AVX)
    
    Returns: (n, dv)
    """
    if Q.dtype == np.float32:
        return attention_f32(Q, K, V, nb_threads, block_size, version)
    elif Q.dtype == np.float64:
        return attention_f64(Q, K, V, nb_threads, block_size, version)
    else:
        raise TypeError("Only float32 and float64 dtypes are supported")