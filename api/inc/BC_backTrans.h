#pragma once

int my_BC_back_trans_v8_10_noBandU(double *Q, long ldQ, double *U, long ldU, long n, int b, cudaStream_t stream = NULL);
