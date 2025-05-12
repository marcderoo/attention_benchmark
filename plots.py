import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des résultats
df = pd.read_csv("results/timings.csv")

# Style graphique
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

# 1. Temps moyen d'exécution
plt.subplot(1, 2, 1)
sns.lineplot(x="dim", y="mean_numpy", data=df, label="NumPy", marker="o")
sns.lineplot(x="dim", y="mean_numba", data=df, label="Numba", marker="s")
sns.lineplot(x="dim", y="mean_cython", data=df, label="Cython", marker="^")
plt.title("Temps moyen d'exécution (en secondes)")
plt.xlabel("Dimension")
plt.ylabel("Temps (s)")
plt.legend()
plt.tight_layout()

# 2. Accélérations
plt.subplot(1, 2, 2)
sns.lineplot(x="dim", y="speedup_numba", data=df, label="Accélération Numba", marker="s")
sns.lineplot(x="dim", y="speedup_cython", data=df, label="Accélération Cython", marker="^")
plt.axhline(1, color='gray', linestyle='--', label="Base NumPy")
plt.title("Accélération par rapport à NumPy")
plt.xlabel("Dimension")
plt.ylabel("Facteur d'accélération")
plt.legend()
plt.tight_layout()

# Affichage
plt.suptitle("Benchmark des implémentations d'attention", fontsize=16, y=1.05)
plt.show()
