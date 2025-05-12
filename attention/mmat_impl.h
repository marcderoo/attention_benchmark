#pragma once

void mmat_impl_cpp(int n_row, int n_col, int k, const float *p1,
                   const float *p2, float *res, int block_size, int version);
void mmat_impl_cpp(int n_row, int n_col, int k, const double *p1,
                   const double *p2, double *res, int block_size, int version);
