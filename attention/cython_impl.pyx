# cython_impl.pyx
import numpy as np
cimport numpy as np
from libc.math cimport expf, sqrtf
from libc.string cimport memset

# Import des instructions AVX pour la vectorisation
cdef extern from "immintrin.h":
    # Types vectoriels AVX
    ctypedef float __m256
    
    # Fonctions AVX
    __m256 _mm256_loadu_ps(const float* mem_addr) nogil
    void _mm256_storeu_ps(float* mem_addr, __m256 a) nogil
    __m256 _mm256_setzero_ps() nogil
    __m256 _mm256_add_ps(__m256 a, __m256 b) nogil
    __m256 _mm256_mul_ps(__m256 a, __m256 b) nogil
    
    # FMA (Fused Multiply-Add)
    __m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c) nogil

# Activer le support OpenMP
cdef extern from "omp.h":
    int omp_get_max_threads() nogil
    int omp_get_num_threads() nogil
    int omp_get_thread_num() nogil
    void omp_set_num_threads(int) nogil

# Constantes pour optimisation
DEF BLOCK_SIZE = 32  # Taille des blocs pour optimiser l'utilisation du cache
DEF AVX_WIDTH = 8    # Nombre de floats traités en parallèle avec AVX (256 bits / 32 bits)

# Fonctions auxiliaires
cdef inline float horizontal_sum(__m256 v) nogil:
    """Somme horizontale d'un vecteur AVX."""
    cdef float result[8]
    _mm256_storeu_ps(result, v)
    return result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7]

# Déclaration de l'interface principale
def attention(np.ndarray[np.float32_t, ndim=2, mode="c"] Q,
              np.ndarray[np.float32_t, ndim=2, mode="c"] K,
              np.ndarray[np.float32_t, ndim=2, mode="c"] V,
              int num_threads=0):
    """
    Compute scaled dot-product attention using Cython with OpenMP and AVX.
    
    Args:
        Q: Query matrix of shape (n, d)
        K: Key matrix of shape (n, d)
        V: Value matrix of shape (n, dk)
        num_threads: Nombre de threads à utiliser (0 = auto)
    
    Returns:
        Output attention matrix of shape (n, dk)
    """
    # Définir le nombre de threads OpenMP
    if num_threads > 0:
        omp_set_num_threads(num_threads)
    
    cdef int n = Q.shape[0]     # Longueur de séquence
    cdef int d = Q.shape[1]     # Dimension de la requête/clé
    cdef int dk = V.shape[1]    # Dimension de la valeur
    cdef float scale = 1.0 / sqrtf(d)
    
    # Allouer mémoire pour résultats et calculs intermédiaires
    cdef np.ndarray[np.float32_t, ndim=2] out = np.zeros((n, dk), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] scores = np.zeros((n, n), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] weights = np.zeros((n, n), dtype=np.float32)
    
    cdef int i, j, k, ii, jj, kk
    cdef int i_block, j_block, k_block
    cdef float max_val, sum_exp
    
    # Pointeurs pour accès direct à la mémoire
    cdef float* q_ptr
    cdef float* k_ptr
    cdef float* s_ptr
    cdef float* v_ptr
    cdef float* w_ptr
    cdef float* o_ptr
    
    # Variables pour vectorisation AVX
    cdef __m256 q_vec, k_vec, sum_vec
    cdef float dot_product
    
    # 1. Calcul des scores d'attention (Q @ K.T) * scale par blocs
    for i_block in range(0, n, BLOCK_SIZE):
        for j_block in range(0, n, BLOCK_SIZE):
            for i in range(i_block, min(i_block + BLOCK_SIZE, n)):
                for j in range(j_block, min(j_block + BLOCK_SIZE, n)):
                    # Calcul vectorisé du produit scalaire
                    dot_product = 0.0
                    k = 0
                    
                    # Pointeurs vers les données
                    q_ptr = &Q[i, 0]
                    k_ptr = &K[j, 0]
                    
                    # Partie vectorisée avec AVX
                    sum_vec = _mm256_setzero_ps()
                    for k in range(0, (d // AVX_WIDTH) * AVX_WIDTH, AVX_WIDTH):
                        q_vec = _mm256_loadu_ps(q_ptr + k)
                        k_vec = _mm256_loadu_ps(k_ptr + k)
                        
                        # Utiliser FMA si disponible, sinon mul + add
                        sum_vec = _mm256_fmadd_ps(q_vec, k_vec, sum_vec)
                    
                    # Somme horizontale du vecteur résultat
                    dot_product = horizontal_sum(sum_vec)
                    
                    # Partie non vectorisée pour le reste
                    for k in range((d // AVX_WIDTH) * AVX_WIDTH, d):
                        dot_product += q_ptr[k] * k_ptr[k]
                    
                    # Appliquer le facteur d'échelle
                    scores[i, j] = dot_product * scale
    
    # 2. Appliquer softmax sur chaque ligne (pour chaque requête)
    for i in range(n):
        # Trouver le maximum pour stabilité numérique
        max_val = scores[i, 0]
        for j in range(1, n):
            if scores[i, j] > max_val:
                max_val = scores[i, j]
        
        # Calculer exp(score - max) et somme
        sum_exp = 0.0
        for j in range(n):
            weights[i, j] = expf(scores[i, j] - max_val)
            sum_exp += weights[i, j]
        
        # Normaliser
        for j in range(n):
            weights[i, j] /= sum_exp
    
    # 3. Calculer la sortie (weights @ V) par blocs
    for i in range(n):
        o_ptr = &out[i, 0]
        
        # Optimisation par blocs pour V
        for j_block in range(0, n, BLOCK_SIZE):
            for k_block in range(0, dk, BLOCK_SIZE):
                for j in range(j_block, min(j_block + BLOCK_SIZE, n)):
                    w_ptr = &weights[i, j]
                    v_ptr = &V[j, 0]
                    
                    # Produit matrice-vecteur optimisé par blocs
                    for k in range(k_block, min(k_block + BLOCK_SIZE, dk)):
                        o_ptr[k] += w_ptr[0] * v_ptr[k]
    
    return out