#pragma once

// #include <cuda.h>
void launchKernel_ClearMatrix(dim3 gridDim, dim3 blockDim, long m, long n, double *A, long ldA);

void launchKernel_setMetrixTrValue(dim3 gridDim, dim3 blockDim, long m, long n, double *A, long ldA, double v);


void launchKernel_CpyMatrixL2U(dim3 gridDim, dim3 blockDim, long n, double *A, long ldA);

void launchKernel_copyAndClear(dim3 gridDim, dim3 blockDim, long m, long n, double *srcM, long lds,
                               double *dstM, long ldd);

void launchKernel_IminusQ(dim3 gridDim, dim3 blockDim, long m, long n, double *Q, long ldq);

void launchKernel_AminusB(dim3 gridDim, dim3 blockDim, long m, long n, double *A, long ldA, double *B, long ldB);

void launchKernel_AbsAminusAbsB(dim3 gridDim, dim3 blockDim, long m, long n, double *A, long ldA, double *B, long ldB);

void launchKernel_copyMatrix(dim3 gridDim, dim3 blockDim, long m, long n, double *srcM, long lds,
                             double *dstM, long ldd);

void launchKernel_copyMatrixAToTranpB(dim3 gridDim, dim3 blockDim, long m, long n, double *srcM, long lds,
                                      double *dstM, long ldd);

void launchKernel_getU(dim3 gridDim, dim3 blockDim, int m, int n, double *A, int ldA, double *U, int ldU);

void launchKernel_getLower(dim3 gridDim, dim3 blockDim, long m, long n, double *A, long ldA);

void launch_kernel_cpyATr2Vector(dim3 gridDim, dim3 blockDim,
                                 long m, long n, double *A, long ldA, double *B);

void launchKernel_scaleMatrixA(dim3 gridDim, dim3 blockDim,
                               long m, long n, double *A, long ldA, double scaler);

double findVectorAbsMax(double *d_array, int n);