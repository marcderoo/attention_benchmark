#include "mmat_impl.h"
#include <immintrin.h>

#include <algorithm>
#include <iostream>
#include <omp.h>

#ifdef __AVX2__
#else
#pragma error "AVX2 not supported"
#endif

// Accès à un élément (i, j)
template <typename DTYPE> inline DTYPE &at(DTYPE *p, int cols, int i, int j) {
  return p[i * cols + j];
}

// Multiplication de matrices plates par blocs
template <typename DTYPE>
void BlockMatrixMultiply(const DTYPE *A, const DTYPE *B, DTYPE *C, int n, int m,
                         int p, int block_size) {
#pragma omp parallel for collapse(2) schedule(dynamic)
  for (int i = 0; i < n; i += block_size) {
    for (int j = 0; j < p; j += block_size) {
      for (int k = 0; k < m; k += block_size) {
        // Sous-bloc
        for (int ii = i; ii < std::min(i + block_size, n); ++ii) {
          for (int jj = j; jj < std::min(j + block_size, p); ++jj) {
            double sum = 0.0;
            for (int kk = k; kk < std::min(k + block_size, m); ++kk) {
              sum += at(A, m, ii, kk) * at(B, p, kk, jj);
            }
            at(C, p, ii, jj) += sum;
          }
        }
      }
    }
  }
}

void BlockMatrixMultiplyAVX(const double *A, const double *B, double *C, int n,
                            int m, int p, int block_size) {
#pragma omp parallel for collapse(2) schedule(dynamic)
  for (int i = 0; i < n; i += block_size) {
    for (int j = 0; j < p; j += block_size) {
      for (int k = 0; k < m; k += block_size) {
        int i_max = std::min(i + block_size, n);
        int j_max = std::min(j + block_size, p);
        int k_max = std::min(k + block_size, m);

        for (int ii = i; ii < i_max; ++ii) {
          int jj = j;
          // Vectorisé par paquets de 4 colonnes
          for (; jj + 3 < j_max; jj += 4) {
            // Charge les 4 valeurs C[ii][jj..jj+3]
            __m256d c_vec = _mm256_loadu_pd(&C[ii * p + jj]);

            // Accumulation sur k
            for (int kk = k; kk < k_max; ++kk) {
              // Broadcast A[ii][kk] dans un vecteur
              __m256d a_vec = _mm256_set1_pd(A[ii * m + kk]);
              // Charge B[kk][jj..jj+3]
              __m256d b_vec = _mm256_loadu_pd(&B[kk * p + jj]);
              // FMA : c_vec += a_vec * b_vec
#ifdef __FMA__
              c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
#else
              c_vec = _mm256_add_pd(_mm256_mul_pd(a_vec, b_vec), c_vec);
#endif              
            }
            // Stocke le résultat
            _mm256_storeu_pd(&C[ii * p + jj], c_vec);
          }
          // Reste scalaire pour les colonnes non alignées
          for (; jj < j_max; ++jj) {
            double sum = 0.0;
            for (int kk = k; kk < k_max; ++kk) {
              sum += A[ii * m + kk] * B[kk * p + jj];
            }
            C[ii * p + jj] += sum;
          }
        }
      }
    }
  }
}

void BlockMatrixMultiplyAVX(const float *A, const float *B, float *C, int n,
                            int m, int p, int block_size) {
#pragma omp parallel for collapse(2) schedule(dynamic)
  for (int i = 0; i < n; i += block_size) {
    for (int j = 0; j < p; j += block_size) {
      for (int k = 0; k < m; k += block_size) {
        int i_max = std::min(i + block_size, n);
        int j_max = std::min(j + block_size, p);
        int k_max = std::min(k + block_size, m);

        for (int ii = i; ii < i_max; ++ii) {
          int jj = j;
          // Vectorisé par paquets de 8 colonnes (float)
          for (; jj + 7 < j_max; jj += 8) {
            // Charge C[ii][jj..jj+7]
            __m256 c_vec = _mm256_loadu_ps(&C[ii * p + jj]);

            // Accumulation sur k
            for (int kk = k; kk < k_max; ++kk) {
              // Broadcast A[ii][kk] dans un vecteur
              __m256 a_vec = _mm256_set1_ps(A[ii * m + kk]);
              // Charge B[kk][jj..jj+7]
              __m256 b_vec = _mm256_loadu_ps(&B[kk * p + jj]);
              // FMA : c_vec += a_vec * b_vec
#ifdef __FMA__
              c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
#else
              c_vec = _mm256_add_ps(_mm256_mul_ps(a_vec, b_vec), c_vec);
#endif              
            }
            // Stocke le résultat
            _mm256_storeu_ps(&C[ii * p + jj], c_vec);
          }
          // Reste scalaire pour les colonnes non multiples de 8
          for (; jj < j_max; ++jj) {
            float sum = 0.0f;
            for (int kk = k; kk < k_max; ++kk) {
              sum += A[ii * m + kk] * B[kk * p + jj];
            }
            C[ii * p + jj] += sum;
          }
        }
      }
    }
  }
}

void mmat_impl_cpp(int n_row, int n_col, int k, const float *p1,
                   const float *p2, float *res, int block_size, int version) {
  if (version == 0)
    BlockMatrixMultiply(p1, p2, res, n_row, k, n_col, block_size);
  else
    BlockMatrixMultiplyAVX(p1, p2, res, n_row, k, n_col, block_size);
}

void mmat_impl_cpp(int n_row, int n_col, int k, const double *p1,
                   const double *p2, double *res, int block_size, int version) {
  if (version == 0)
    BlockMatrixMultiply(p1, p2, res, n_row, k, n_col, block_size);
  else
    BlockMatrixMultiplyAVX(p1, p2, res, n_row, k, n_col, block_size);
}
