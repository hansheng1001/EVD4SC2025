#pragma once
#include <cusolverDn.h>

void my_ZY_ZY_SBR(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle,
                  long M, long N, long b, long nb, double *dOriA, long ldOriA2, double *dA, long ldA,
                  double *dW, long ldW, double *dY, long ldY, double *dZ, long ldZ, double *dR, long ldR, double *work, int *info);

void my_ZY_ZY_SBR_V2(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle,
                     long M, long N, long b, long nb, double *dOriA, long ldOriA2, double *dA, long ldA,
                     double *dW, long ldW, double *dY, long ldY, double *dZ, long ldZ, double *dR, long ldR, double *work, int *info);
