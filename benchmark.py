import os
import platform
import time
import numpy as np
import pandas as pd
from statistics import mean, stdev

from rich.console import Console
from rich.table import Table
from rich import box

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
NUM_THREADS = 1

# (1) Limitation des threads pour OpenBLAS/MKL/NumPy
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS)

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
def find_best_combo(Q, K, V, block_sizes, thread_values, dtypes, warmup=3, repeat_init=5, top_k=3, final_repeat=15):
    import heapq
    perf = []
    
    for dtype in dtypes:
        Q = Q.astype(dtype)
        K = K.astype(dtype)
        V = V.astype(dtype)

        for bs in block_sizes:
            for nt in thread_values:
                try:
                    times = measure(attention_cython, (Q, K, V, nt, bs), warmup=warmup, repeat=repeat_init)
                    avg_time = mean(times)
                    perf.append((avg_time, nt, bs, dtype))
                except Exception as e:
                    print(f"[WARNING] Combo (bs={bs}, nt={nt}, dtype={dtype}) échoue : {e}")

    if not perf:
        raise RuntimeError("Aucune combinaison (block_size, nb_threads, dtype) valide.")

    # On garde top_k combinaisons
    top_candidates = heapq.nsmallest(top_k, perf)
    selected = [(nt, bs, dtype) for _, nt, bs, dtype in top_candidates]

    final_results = []
    for nt, bs, dtype in selected:
        Q = Q.astype(dtype)
        K = K.astype(dtype)
        V = V.astype(dtype)
        
        try:
            times = measure(attention_cython, (Q, K, V, nt, bs), warmup=warmup, repeat=final_repeat)
            avg_time = mean(times)
            final_results.append((avg_time, nt, bs, dtype))
        except Exception as e:
            print(f"[WARNING] Combo finale échoue : {e}")

    best_time, best_nt, best_bs, best_dtype = min(final_results)
    print(f"[INFO] → combo optimal : block_size={best_bs}, nb_threads={best_nt}, dtype={best_dtype} (temps moyen : {best_time:.6f} s)")
    return best_nt, best_bs, best_dtype


# ---------------------------------------------------
# 5. Exécution du benchmark 
# ---------------------------------------------------
def run_benchmark(dims=None, block_sizes=None, thread_values=None, repeat: int = 50, warmup: int = 5, output_dir: str = "results"):
    if dims is None:
        dims = [64, 128, 256, 400, 512, 768, 1024]

    if thread_values is None:
        thread_values = [1, 2, 4, 8]

    if block_sizes is None:
        block_sizes = [8, 16, 32, 64]

    os.makedirs(output_dir, exist_ok=True)

    dtypes = [np.float32, np.float64]
    
    data = []

    for dim in dims:
        print(f"Benchmark pour dim={dim}")
        
        # Initial Q, K, V matrices with float32 for initial testing
        Q = np.random.rand(dim, dim).astype(np.float32)
        K = np.random.rand(dim, dim).astype(np.float32)
        V = np.random.rand(dim, dim).astype(np.float32)

        # Find the best combination of block_size, nb_threads, and dtype
        best_nb_threads, best_block_size, best_dtype = find_best_combo(Q, K, V, block_sizes, thread_values, dtypes, warmup, repeat)
        
        # Now create matrices with the best dtype found
        Q = np.random.rand(dim, dim).astype(best_dtype)
        K = np.random.rand(dim, dim).astype(best_dtype)
        V = np.random.rand(dim, dim).astype(best_dtype)
        
        # Measure performances with the selected best parameters
        times_np = measure(attention_numpy, (Q, K, V), warmup, repeat)
        times_nb = measure(attention_numba, (Q, K, V), warmup, repeat)

        times_cy = measure(attention_cython, (Q, K, V, best_nb_threads, best_block_size), warmup, repeat)

        try:
            all_close = (
                np.allclose(attention_numpy(Q, K, V), attention_numba(Q, K, V), atol=1e-5, rtol=1e-3)
                and np.allclose(attention_numpy(Q, K, V), attention_cython(Q, K, V, best_nb_threads, best_block_size), atol=1e-5, rtol=1e-3)
            )
        except Exception as e:
            all_close = False
            print(f"[WARNING] Erreur lors de la vérification all_close: {e}")

        record = {
            "dim": dim,
            "dtype": str(best_dtype).replace("<class '", "").replace("'>", ""),
            "block_size_cython": best_block_size,
            "best_nb_threads": best_nb_threads,
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

    # Sauvegarde CSV complet
    df = pd.DataFrame(data)
    output_csv = os.path.join(output_dir, f"timings_summary.csv")
    df.to_csv(output_csv, index=False)
    print(f"\n[INFO] Résultats enregistrés dans {output_csv}")

    # Création des colonnes formatées avec notation scientifique et ± std
    for method in ['numpy', 'numba', 'cython']:
        df[f"{method.capitalize()}"] = df.apply(
            lambda row: f"{row[f'mean_{method}']:.2e}", axis=1
        )

    # Optionnel : suppression des colonnes séparées mean/stdev
    df_display = df.drop(columns=[
        "mean_numpy", "stdev_numpy",
        "mean_numba", "stdev_numba",
        "mean_cython", "stdev_cython"
    ])

    # Réorganisation optionnelle des colonnes pour l’affichage
    cols = ['dim', 'dtype', 'block_size_cython', 'best_nb_threads',
            'Numpy', 'Numba', 'Cython',
            'speedup_numba', 'speedup_cython', 'all_close']

    df_display = df_display[cols]

    # Crée une console Rich
    console = Console()

    # Crée une table Rich avec bordures arrondies
    table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE_HEAVY)

    # Ajoute les colonnes avec couleur personnalisée pour certaines
    for col in df_display.columns:
        if col in ['Numpy', 'Numba', 'Cython']:
            table.add_column(col, style="cyan", justify="right")
        elif col.startswith('speedup'):
            table.add_column(" ".join([col_.capitalize() for col_ in col.split("_")]), style="green", justify="right")
        elif col == 'all_close':
            table.add_column(" ".join([col_.capitalize() for col_ in col.split("_")]), style="bold red", justify="center")
        else:
            table.add_column(" ".join([col_.capitalize() for col_ in col.split("_")]), justify="center")

    # Ajoute les lignes une à une
    for _, row in df_display.iterrows():
        row_values = []
        for col in df_display.columns:
            val = row[col]
            # Format spécifique pour speedup_numba
            if col.startswith('speedup'):
                val = f"{val:.3f}"
            row_values.append(str(val))
        table.add_row(*row_values)

    # Affiche le tableau dans la console
    console.print(table)

# ---------------------------------------------------
# 6. Point d'entrée
# ---------------------------------------------------
if __name__ == "__main__":
    run_benchmark()