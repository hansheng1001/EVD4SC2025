#pragma once
#include <cusolverDn.h>


template <typename T>
void my_ZY_ZY_SBR_Vector(cusolverDnHandle_t cusolver_handle,
                  cublasHandle_t cublas_handle,
                  long M,
                  long N,
                  long b,
                  long nb,
                  T *dOriA,
                  long ldOriA2,
                  T *dA,
                  long ldA,
                  T *dW,
                  long ldW,
                  T *dY,
                  long ldY,
                  T *dZ,
                  long ldZ,
                  T *dR,
                  long ldR,
                  T *work,
                  int *info);
