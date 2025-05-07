import numpy as np
cimport numpy as np


# Déclaration de la fonction externe en C++ depuis le fichier .h
cdef extern from "attention_impl.h":
    void attention_impl_cpp(int n_row, int n_col, int k,
                             const float* p1, const float* p2, float* res,
                             int block_size,
                             int version)
    void attention_impl_cpp(int n_row, int n_col, int k,
                             const double* p1, const double* p2, double* res,
                             int block_size, int version)


# Fonctions Cython pour gérer la multiplication
cdef attention_c_float(const float[:, ::1] a, const float [:, ::1] b, float [:, ::1] res,
                       int block_size, int version):
    attention_impl_cpp(a.shape[0], b.shape[1], a.shape[0], &a[0, 0], &b[0, 0], &res[0, 0],
                       block_size, version)


cdef attention_c_double(const double[:, ::1] a, const double[:, ::1] b, double[:, ::1] res,
                        int block_size, int version):
    attention_impl_cpp(a.shape[0], b.shape[1], a.shape[0], &a[0, 0], &b[0, 0], &res[0, 0],
                       block_size, version)


# Fonction principale d'appel en Python
def _attention(np.ndarray a, np.ndarray b, block_size=16, version=0):
    res = np.zeros((a.shape[0], b.shape[1]), dtype=a.dtype)

    if a.dtype == np.float32:
        attention_c_float(a, b, res, block_size, version)
    elif a.dtype == np.float64:
        attention_c_double(a, b, res, block_size, version)
    else:
        raise NotImplementedError(f"Not implemented for dtype={a.dtype}")
    return res


def attention(a, b, block_size=16, version=0):
    """Attention mechanism simulation."""
    assert len(a.shape) == 2 == len(b.shape), (
        f"Only applies on matrices but a.shape={a.shape}, b.shape={b.shape}"
    )
    assert a.shape[1] == b.shape[0], (
        f"Shape mismatch: a.shape={a.shape}, b.shape={b.shape}"
    )
    assert a.dtype == b.dtype, f"Type mismatch a.dtype={a.dtype}, b.dtype={b.dtype}"
    assert a.flags["C_CONTIGUOUS"], "Matrix a must be contiguous"
    assert b.flags["C_CONTIGUOUS"], "Matrix b must be contiguous"

    return _attention(a, b, block_size, version)
