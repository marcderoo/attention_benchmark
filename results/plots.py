import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style and color palette
sns.set(style="whitegrid", palette="muted")

for dtype in ["float32", "float64"]:
    # Load the results
    df = pd.read_csv(f"results/timings_{dtype}.csv")

    # Create the figure with subplots
    plt.figure(figsize=(18, 8))

    # 1. Execution Time (Mean Times for NumPy, Numba, Cython)
    plt.subplot(2, 2, 1)
    sns.lineplot(x="dim", y="mean_numpy", data=df, label="NumPy", marker="o", linewidth=2)
    sns.lineplot(x="dim", y="mean_numba", data=df, label="Numba", marker="s", linewidth=2)
    sns.lineplot(x="dim", y="mean_cython", data=df, label="Cython", marker="^", linewidth=2)
    plt.title(f"Mean Execution Time for {dtype}", fontsize=14)
    plt.xlabel("Dimension", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.legend(title="Implementations", loc="upper left")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    # 2. Speedup (Comparing Numba and Cython with NumPy as baseline)
    plt.subplot(2, 2, 2)
    sns.lineplot(x="dim", y="speedup_numba", data=df, label="Speedup Numba", marker="s", linewidth=2)
    sns.lineplot(x="dim", y="speedup_cython", data=df, label="Speedup Cython", marker="^", linewidth=2)
    plt.axhline(1, color='gray', linestyle='--', label="Base NumPy")
    plt.title(f"Speedup Relative to NumPy ({dtype})", fontsize=14)
    plt.xlabel("Dimension", fontsize=12)
    plt.ylabel("Speedup Factor", fontsize=12)
    plt.legend(title="Speedup", loc="upper left")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.yscale('log')  # Use log scale for better visualization of speedup
    
    # 3. Best Block Size Impact (Show best block size for each dimension)
    plt.subplot(2, 2, 3)
    sns.lineplot(x="dim", y="best_block_size", data=df, label="Best Block Size", marker="o", color="green", linewidth=2)
    plt.title(f"Optimal Block Size for {dtype}", fontsize=14)
    plt.xlabel("Dimension", fontsize=12)
    plt.ylabel("Block Size", fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # 4. Comparison of Stdev (Variability of Each Implementation)
    plt.subplot(2, 2, 4)
    sns.lineplot(x="dim", y="stdev_numpy", data=df, label="Stdev NumPy", marker="o", linewidth=2)
    sns.lineplot(x="dim", y="stdev_numba", data=df, label="Stdev Numba", marker="s", linewidth=2)
    sns.lineplot(x="dim", y="stdev_cython", data=df, label="Stdev Cython", marker="^", linewidth=2)
    plt.title(f"Execution Time Variability for {dtype}", fontsize=14)
    plt.xlabel("Dimension", fontsize=12)
    plt.ylabel("Standard Deviation (seconds)", fontsize=12)
    plt.legend(title="Standard Deviation", loc="upper left")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Overall title and layout
    plt.suptitle(f"Benchmark of Attention Implementations - {dtype}", fontsize=16, y=1.03)
    plt.tight_layout()
    plt.show()
