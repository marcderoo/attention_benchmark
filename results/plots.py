import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load summary CSV
df = pd.read_csv("results/timings_summary.csv")

# Set seaborn style
sns.set(style="whitegrid", palette="colorblind", font_scale=1.1)

# Iterate over float32 and float64
for dtype in df["dtype"].unique():
    df_dtype = df[df["dtype"] == dtype].copy()

    # Compute coefficient of variation for each implementation
    df_dtype["cv_numpy"]  = df_dtype["stdev_numpy"]  / df_dtype["mean_numpy"]
    df_dtype["cv_numba"]  = df_dtype["stdev_numba"]  / df_dtype["mean_numba"]
    df_dtype["cv_cython"] = df_dtype["stdev_cython"] / df_dtype["mean_cython"]

    # Create the figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(f"Benchmark of Attention Implementations - {dtype}", fontsize=18, y=1.03)

    # Plot 1 - Execution Times
    ax1 = axes[0, 0]
    sns.lineplot(ax=ax1, x="dim", y="mean_numpy", data=df_dtype, label="NumPy", marker="o")
    sns.lineplot(ax=ax1, x="dim", y="mean_numba", data=df_dtype, label="Numba", marker="s")
    sns.lineplot(ax=ax1, x="dim", y="mean_cython", data=df_dtype, label="Cython", marker="^")
    ax1.set_title("Mean Execution Time")
    ax1.set_xlabel("Dimension")
    ax1.set_ylabel("Time (seconds)")
    ax1.legend()
    ax1.grid(True, linestyle="--", linewidth=0.5)

    # Plot 2 - Speedups
    ax2 = axes[0, 1]
    sns.lineplot(ax=ax2, x="dim", y="speedup_numba", data=df_dtype, label="Speedup Numba", marker="s")
    sns.lineplot(ax=ax2, x="dim", y="speedup_cython", data=df_dtype, label="Speedup Cython", marker="^")
    ax2.axhline(1, color='gray', linestyle='--', label="Baseline (NumPy)")
    ax2.set_title("Speedup Over NumPy")
    ax2.set_xlabel("Dimension")
    ax2.set_ylabel("Speedup Factor (log scale)")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, linestyle="--", linewidth=0.5)

    # Plot 3 - Best Block Size and Threads
    ax3 = axes[1, 0]
    sns.lineplot(ax=ax3, x="dim", y="block_size_cython", data=df_dtype, label="Best Block Size", marker="o", color="darkgreen")
    sns.lineplot(ax=ax3, x="dim", y="best_nb_threads", data=df_dtype, label="Best Thread Count", marker="D", color="darkred")
    ax3.set_title("Best Block Size and Thread Count")
    ax3.set_xlabel("Dimension")
    ax3.set_ylabel("Value")
    ax3.legend()
    ax3.grid(True, linestyle="--", linewidth=0.5)

    # Plot 4 - Coefficient of Variation
    ax4 = axes[1, 1]
    sns.lineplot(ax=ax4, x="dim", y="cv_numpy", data=df_dtype, label="CV NumPy", marker="o")
    sns.lineplot(ax=ax4, x="dim", y="cv_numba", data=df_dtype, label="CV Numba", marker="s")
    sns.lineplot(ax=ax4, x="dim", y="cv_cython", data=df_dtype, label="CV Cython", marker="^")
    ax4.set_title("Coefficient of Variation (std / mean)")
    ax4.set_xlabel("Dimension")
    ax4.set_ylabel("Relative Variability")
    ax4.legend()
    ax4.grid(True, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # leave room for suptitle
    plt.show()