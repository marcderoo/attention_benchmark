import numpy as np
import numba
from numba import njit, prange

@njit(parallel=True)
def softmax_2d(x, parallel: bool):
    n, m = x.shape
    out = np.empty((n, m), dtype=x.dtype)
    if parallel:
        for i in prange(n):
            row = x[i]
            max_val = np.max(row)
            exps = np.exp(row - max_val)
            out[i] = exps / np.sum(exps)
    else:
        for i in range(n):
            row = x[i]
            max_val = np.max(row)
            exps = np.exp(row - max_val)
            out[i] = exps / np.sum(exps)
    return out

@njit(parallel=True)
def matmul_blocked(A, B, block_size: int, parallel: bool):
    n, m = A.shape
    m2, p = B.shape
    assert m == m2
    C = np.zeros((n, p), dtype=A.dtype)
    if parallel:
        i0_iter = prange(0, n, block_size)
    else:
        i0_iter = range(0, n, block_size)
    for i0 in i0_iter:
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

@njit(parallel=True)
def _attention_impl(Q, K, V, parallel: bool, block_size: int):
    # type: (np.ndarray, np.ndarray, np.ndarray, bool, int) -> np.ndarray
    # ensure consistent dtype
    dtype = Q.dtype
    Qc = Q.astype(dtype)
    Kc = K.astype(dtype)
    Vc = V.astype(dtype)
    d = dtype.type(Qc.shape[1])

    # compute scaled dot-product
    scores = matmul_blocked(Qc, Kc.T, block_size, parallel) / np.sqrt(d)
    weights = softmax_2d(scores, parallel)
    return matmul_blocked(weights, Vc, block_size, parallel)


def attention(Q: np.ndarray,
              K: np.ndarray,
              V: np.ndarray,
              num_threads: int = 1,
              block_size: int = 32) -> np.ndarray:
    """
    Compute attention: softmax(QK^T / sqrt(d)) V

    Parameters
    ----------
    Q, K, V : np.ndarray
        Input query, key, value matrices (shape [n, d]).
    num_threads : int, optional
        Number of threads to use
    block_size : int, optional
        Block size for the blocked matmul, by default 32.

    Returns
    -------
    np.ndarray
        The attention output matrix.
    """
    # configure Numba threading
    if num_threads > 1:
        numba.set_num_threads(num_threads)
    else:
        numba.set_num_threads(1)

    # call compiled implementation
    return _attention_impl(Q, K, V, num_threads > 1, block_size)