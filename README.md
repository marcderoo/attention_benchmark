# attention_benchmark

Projet programmation parallèle 

**TO DO**
- se renseigner pour mieux stabiliser les résultats.
- améliorer cython
- améliorer benchmark :
   -ajouter hyperparamètres type block size, nb_thread aux fonctions
   - fonction recherche active des meilleurs hyperparamètres
- faire des beux plots des perfs et choix hyperparamètres
- rédiger rapport

**But**: 
  - battre/ s’approcher de Numpy sur le calcul de l’attention en utilisant avx(paralleliser)
  - optimiser le benchmark càd trouver les meilleurs hyper paramètres (par ex blocksize, version en le moins d’essais possible).
  - il faut obtenir le même résultat à chaque run.

Rendu rapport (2 pages avec graphique) + code date de rendu 23/05

S'inspirer du prof : https://github.com/sdpython/teachcompute/tree/main/_tutoriels/cython_mat
Cours : https://sdpython.github.io/doc/teachcompute/dev/articles/2025-05-31-route2025.html#points-particuliers

**Steps**
- Implement and benchmark the scaled dot-product attention
- Implement this with NumPy as baseline and one or more parallelized versions (Numba, Cython).
- Build a benchmark function similar to your matrix multiplication setup.
- Start with simple hyperparameter tuning, e.g.: block_size / parallel version (e.g., naive vs. tiled)
- Gradually move to adaptive search to minimize the number of trials needed.



```
attention-benchmark/
├── README.md
├── requirements.txt
├── setup.py
├── benchmark.py                 # Main benchmark script
├── run.py                       # Entry point for launching benchmarks / tuning
├── attention/                   # Package folder
│   ├── __init__.py
│   ├── numpy_impl.py            # NumPy reference implementation
│   ├── numba_impl.py            # Numba-optimized implementation
│   ├── cython_impl.pyx          # Cython interface
│   ├── cython_impl.cpp          # C++ implementation
│   ├── cython_impl.h            # Header for the C++ code
├── results/                     # Folder to save output CSVs, plots, etc.
│   ├── timings.csv
│   └── plots/
├── .gitignore
```
