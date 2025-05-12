import os
import platform
import time
import numpy as np
import pandas as pd
from statistics import mean, stdev

# Ajout de psutil et resource pour gestion CPU/mémoire
try:
    import psutil
except ImportError:
    psutil = None

try:
    import resource  # Unix only
except ImportError:
    resource = None

# ---------------------------------------------------
# 1. Environnement et reproductibilité
# ---------------------------------------------------
np.random.seed(42)

# (1) Limitation des threads pour OpenBLAS/MKL/NumPy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# (2) Limitation de l’affinité CPU à 0-3
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

# (3) Limitation mémoire (RAM)
MAX_RAM_MB = 2048  # Exemple : 2 Go
max_bytes = MAX_RAM_MB * 1024 ** 2

if platform.system() in ("Linux", "Darwin") and resource:
    # Unix (Linux/macOS)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, hard))
        print(f"[INFO] Limite de mémoire fixée à {MAX_RAM_MB} MB (Unix).")
    except ValueError:
        print("[WARNING] Impossible de fixer la limite mémoire avec resource.")
elif platform.system() == "Windows":
    # Windows via Job Object (nécessite pywin32)
    try:
        import win32job
        import win32process
        import win32con

        # création du Job Object
        job = win32job.CreateJobObject(None, "")
        info = win32job.QueryInformationJobObject(
            job, win32job.JobObjectExtendedLimitInformation
        )
        # drapeaux et limite mémoire
        info['BasicLimitInformation']['LimitFlags'] = (
            win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
        )
        info['ProcessMemoryLimit'] = max_bytes
        win32job.SetInformationJobObject(
            job,
            win32job.JobObjectExtendedLimitInformation,
            info
        )
        # assignation du processus courant au Job
        hProcess = win32process.GetCurrentProcess()
        win32job.AssignProcessToJobObject(job, hProcess)
        print(f"[INFO] Limite de mémoire fixée à {MAX_RAM_MB} MB (Windows).")
    except ImportError:
        print("[WARNING] Pour limiter la RAM sur Windows, installez pywin32 (`pip install pywin32`).")
    except Exception as e:
        print(f"[WARNING] Impossible de fixer la limite mémoire sur Windows : {e}")
else:
    print("[WARNING] Limitation mémoire non supportée sur ce système.")

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
    for _ in range(warmup):
        fn(*args)
    durations = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        durations.append(t1 - t0)
    return durations


# --------------------------------------------------------------------------
# 4. Recherche dynamique (successive halving) des meilleurs hyperparamètres
# --------------------------------------------------------------------------
def find_best_block_size(Q, K, V, block_sizes, warmup=3, repeat_init=5, top_k=3, final_repeat=30):
    """
    Trouve dynamiquement le meilleur block_size pour attention_cython en minimisant le temps moyen.
    
    Étapes :
    - Bench initial rapide pour chaque block_size
    - On garde les top_k plus rapides
    - On refait un bench plus long et on choisit le meilleur

    Retourne : block_size optimal
    """
    import heapq

    # Étape 1 : Benchmark initial rapide
    perf = []
    for bs in block_sizes:
        try:
            times = measure(attention_cython, (Q, K, V, bs), warmup=warmup, repeat=repeat_init)
            avg_time = mean(times)
            perf.append((avg_time, bs))
        except Exception as e:
            print(f"[WARNING] Block size {bs} échoue : {e}")

    if not perf:
        raise RuntimeError("Aucune valeur de block_size valide trouvée.")

    # Étape 2 : Garde top_k meilleurs
    top_candidates = heapq.nsmallest(top_k, perf)
    selected = [bs for _, bs in top_candidates]

    # Étape 3 : Benchmark final plus précis
    final_results = []
    for bs in selected:
        try:
            times = measure(attention_cython, (Q, K, V, bs), warmup=warmup, repeat=final_repeat)
            avg_time = mean(times)
            final_results.append((avg_time, bs))
        except Exception as e:
            print(f"[WARNING] Block size {bs} échoue au benchmark final : {e}")

    if not final_results:
        raise RuntimeError("Échec du benchmark final pour toutes les valeurs sélectionnées.")

    best_time, best_bs = min(final_results)
    print(f"[INFO] → block_size optimal trouvé : {best_bs} (temps moyen : {best_time:.6f} s)")
    return best_bs


# ---------------------------------------------------
# 5. Exécution du benchmark modifiée pour float32/64
# ---------------------------------------------------
def run_benchmark(dims=None, block_sizes=None, repeat: int = 50, warmup: int = 5, output_dir: str = "results"):
    if dims is None:
        dims = [64, 128, 256, 400, 512, 768, 1024]

    if block_sizes is None:
        block_sizes = [8, 16, 32, 64]

    os.makedirs(output_dir, exist_ok=True)

    for dtype in [np.float32, np.float64]:
        dtype_name = "float32" if dtype == np.float32 else "float64"
        data = []

        print(f"\n[INFO] Démarrage benchmark pour dtype = {dtype_name}\n")

        for dim in dims:
            print(f"Benchmark pour dim={dim} ({dtype_name})")

            Q = np.random.rand(dim, dim).astype(dtype)
            K = np.random.rand(dim, dim).astype(dtype)
            V = np.random.rand(dim, dim).astype(dtype)

            # Mesure numpy et numba (sans block_size)
            times_np = measure(attention_numpy, (Q, K, V), warmup, repeat)
            times_nb = measure(attention_numba, (Q, K, V), warmup, repeat)

            # Recherche du meilleur block_size pour cython
            best_block_size = find_best_block_size(Q, K, V, block_sizes, warmup=3, repeat_init=5, top_k=2, final_repeat=15)
            times_cy = measure(attention_cython, (Q, K, V, 0, best_block_size), warmup, repeat)

            try:
                all_close = (
                    np.allclose(attention_numpy(Q, K, V), attention_numba(Q, K, V), atol=1e-5, rtol=1e-3)
                    and np.allclose(attention_numpy(Q, K, V), attention_cython(Q, K, V, 0, best_block_size), atol=1e-5, rtol=1e-3)
                )
            except Exception as e:
                all_close = False
                print(f"[WARNING] Erreur lors de la vérification all_close: {e}")

            record = {
                "dim": dim,
                "dtype": dtype_name,
                "best_block_size": best_block_size,
                "mean_numpy": mean(times_np),
                "stdev_numpy": stdev(times_np),
                "mean_numba": mean(times_nb),
                "stdev_numba": stdev(times_nb),
                "mean_cython": mean(times_cy),
                "stdev_cython": stdev(times_cy),
                "speedup_numba": mean(times_np) / mean(times_nb),
                "speedup_cython": mean(times_np) / mean(times_cy),
                "all_close": all_close
            }
            data.append(record)

        df = pd.DataFrame(data)
        output_csv = os.path.join(output_dir, f"timings_{dtype_name}.csv")
        df.to_csv(output_csv, index=False)
        print(f"\n[INFO] Résultats ({dtype_name}) enregistrés dans {output_csv}")
        print(df)


# ---------------------------------------------------
# 6. Point d'entrée
# ---------------------------------------------------
if __name__ == "__main__":
    run_benchmark()