
#pragma once

#include <cusolverDn.h>

void panelQR(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle,
             long m, long n, double *A, long lda, double *W, long ldw,
             double *R, long ldr, double *work, int *info);
