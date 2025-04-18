
#pragma once

#include <cusolverDn.h>


template <typename T>
void panelQR(cusolverDnHandle_t cusolver_handle,
             cublasHandle_t cublas_handle,
             long m,
             long n,
             T *A,
             long lda,
             T *W,
             long ldw,
             T *R,
             long ldr,
             T *work,
             int *info);
