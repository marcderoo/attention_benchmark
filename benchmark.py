import numpy as np
import pandas as pd
import time

from attention.numpy_impl import attention as attention_numpy
from attention.numba_impl import attention as attention_numba
from attention.cython_impl import attention as attention_cython


def bench(name, fn, Q, K, V, repeat=20, warmup=3):
    for i in range(repeat):
        if i == warmup:
            start = time.time()
        fn(Q, K, V)
    return time.time() - start

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
        data.append({
            "dim": dim,
            "t_numpy": t_numpy,
            "t_numba": t_numba,
            "t_cython": t_cython,
            "speedup_numba": t_numpy / t_numba,
            "speedup_cython": t_numpy / t_cython,
        })

    df = pd.DataFrame(data)
    print(df)
    df.to_csv("results/timings.csv", index=False)

if __name__ == "__main__":
    run_benchmark()
