import numpy as np
import pandas as pd
import time

# Set the seed
np.random.seed(42)

from attention.numpy_impl import attention as attention_numpy
from attention.numba_impl import attention as attention_numba
from attention.cython_impl import attention as attention_cython

def bench(name, fn, Q, K, V, repeat=20, warmup=3):
    for i in range(repeat):
        if i == warmup:
            start = time.time()
        fn(Q, K, V)
    return time.time() - start

def test_correctness(Q, K, V, atol=1e-5, rtol=1e-3):
    out_numpy = attention_numpy(Q, K, V)
    out_numba = attention_numba(Q, K, V)
    out_cython = attention_cython(Q, K, V)

    return (
        np.allclose(out_numpy, out_numba, atol=atol, rtol=rtol) and
        np.allclose(out_numpy, out_cython, atol=atol, rtol=rtol)
    )

def run_benchmark():
    dims = [64, 128, 256, 400, 512, 768, 1024]
    repeat = 20
    data = []

    for dim in dims:
        print(f"Testing dim={dim}")
        Q = np.random.rand(dim, dim).astype(np.float32)
        K = np.random.rand(dim, dim).astype(np.float32)
        V = np.random.rand(dim, dim).astype(np.float32)

        t_numpy = bench("numpy", attention_numpy, Q, K, V, repeat)
        t_numba = bench("numba", attention_numba, Q, K, V, repeat)
        t_cython = bench("cython", attention_cython, Q, K, V, repeat)

        correctness = test_correctness(Q, K, V)

        data.append({
            "dim": dim,
            "t_numpy": t_numpy,
            "t_numba": t_numba,
            "t_cython": t_cython,
            "speedup_numba": t_numpy / t_numba,
            "speedup_cython": t_numpy / t_cython,
            "all_close": correctness
        })

    df = pd.DataFrame(data)
    print(df)
    df.to_csv("results/timings.csv", index=False)

if __name__ == "__main__":
    run_benchmark()
