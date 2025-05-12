# attention_benchmark

Projet programmation parallèle 

**TO DO**
- améliorer benchmark :
   - ajouter nb_thread ?
   - sur numba : on peut le modifier pour qu'il parallelise et qu'il prenne en compte block_size. Necessaire ? jsp.. il est déjà bon !
- faire des beaux plots des perfs et choix hyperparamètres
- rédiger rapport

**But**: 
  - battre/ s’approcher de Numpy sur le calcul de l’attention en utilisant avx(paralleliser)
  - optimiser le benchmark càd trouver les meilleurs hyper paramètres (par ex blocksize, version en le moins d’essais possible).
  - il faut obtenir le même résultat à chaque run.

Rendu rapport (2 pages avec graphique) + code date de rendu 23/05


- S'inspirer du prof : https://github.com/sdpython/teachcompute/tree/main/_tutoriels/cython_mat
- Cours : https://sdpython.github.io/doc/teachcompute/dev/articles/2025-05-31-route2025.html#points-particuliers


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
