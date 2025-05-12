import os
import platform
import time
import numpy as np
import pandas as pd
from statistics import mean, stdev, median

# Ajout de psutil pour affinité CPU sous Windows
try:
    import psutil
except ImportError:
    psutil = None

# ---------------------------------------------------
# 1. Environnement et reproductibilité
# ---------------------------------------------------
np.random.seed(42)

# Optionnel : pinning CPU sur cœurs 0-3
if platform.system() == "Linux":
    os.system("taskset -c 0-3 true")
elif platform.system() == "Windows" and psutil:
    try:
        p = psutil.Process()
        p.cpu_affinity([0, 1, 2, 3])
        print("[INFO] Affinité CPU fixée à 0-3 avec psutil.")
    except Exception as e:
        print(f"[WARNING] Impossible de fixer l'affinité CPU : {e}")
elif platform.system() == "Windows":
    print("[INFO] Pour fixer l'affinité CPU sur Windows, installez psutil ou utilisez le Gestionnaire des tâches.")

# ---------------------------------------------------
# 2. Implémentations d'attention
# ---------------------------------------------------
from attention.numpy_impl import attention as attention_numpy
from attention.numba_impl import attention as attention_numba
from attention.cython_impl import attention as attention_cython

# ---------------------------------------------------
# 3. Fonction de mesure
# ---------------------------------------------------
def measure(fn, args, warmup: int = 5, repeat: int = 50):
    """
    Mesure la durée d'exécution de fn(*args) en secondes.
    - warmup: nombre d'appels avant mesure
    - repeat: nombre de répétitions mesurées
    Retourne la liste des durées.
    """
    for _ in range(warmup):
        fn(*args)

    durations = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        durations.append(t1 - t0)
    return durations

# ---------------------------------------------------
# 4. Exécution du benchmark
# ---------------------------------------------------
def run_benchmark(dims=None, repeat: int = 50, warmup: int = 5, output_csv: str = "results/timings.csv"):
    if dims is None:
        dims = [64, 128, 256, 400, 512, 768, 1024]

    data = []
    for dim in dims:
        print(f"Benchmark pour dim={dim}")
        Q = np.random.rand(dim, dim).astype(np.float32)
        K = np.random.rand(dim, dim).astype(np.float32)
        V = np.random.rand(dim, dim).astype(np.float32)

        times_np = measure(attention_numpy, (Q, K, V), warmup, repeat)
        times_nb = measure(attention_numba, (Q, K, V), warmup, repeat)
        times_cy = measure(attention_cython, (Q, K, V), warmup, repeat)

        record = {
            "dim": dim,
            "mean_numpy": mean(times_np),
            "stdev_numpy": stdev(times_np),
            "median_numpy": median(times_np),
            "p95_numpy": sorted(times_np)[int(0.95 * len(times_np))],
            "mean_numba": mean(times_nb),
            "stdev_numba": stdev(times_nb),
            "median_numba": median(times_nb),
            "p95_numba": sorted(times_nb)[int(0.95 * len(times_nb))],
            "mean_cython": mean(times_cy),
            "stdev_cython": stdev(times_cy),
            "median_cython": median(times_cy),
            "p95_cython": sorted(times_cy)[int(0.95 * len(times_cy))],
            "speedup_numba": mean(times_np) / mean(times_nb),
            "speedup_cython": mean(times_np) / mean(times_cy),
            "all_close": (
                np.allclose(attention_numpy(Q, K, V), attention_numba(Q, K, V), atol=1e-5, rtol=1e-3)
                and np.allclose(attention_numpy(Q, K, V), attention_cython(Q, K, V), atol=1e-5, rtol=1e-3)
            )
        }
        data.append(record)

    df = pd.DataFrame(data)
    print(df)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Résultats enregistrés dans {output_csv}")

# ---------------------------------------------------
# 5. Point d'entrée
# ---------------------------------------------------
if __name__ == "__main__":
    run_benchmark()