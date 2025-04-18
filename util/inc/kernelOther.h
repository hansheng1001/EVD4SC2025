#pragma once



template <typename T>
void launchKernel_ClearMatrix(dim3 gridDim, dim3 blockDim, long m, long n, T *A, long ldA, cudaStream_t stream = NULL);



template <typename T>
void launchKernel_setMetrixTrValue(dim3 gridDim, dim3 blockDim, long m, long n, T *A, long ldA, T v, cudaStream_t stream = NULL);


template <typename T>
void launchKernel_CpyMatrixL2U(dim3 gridDim, dim3 blockDim, long n, T *A, long ldA);


template <typename T>
void launchKernel_copyAndClear(dim3 gridDim,
                               dim3 blockDim,
                               long m,
                               long n,
                               T *srcM,
                               long lds,
                               T *dstM,
                               long ldd);



template <typename T>
void launchKernel_IminusQ(dim3 gridDim, dim3 blockDim, long m, long n, T *Q, long ldq);

void launchKernel_AminusB(dim3 gridDim,
                          dim3 blockDim,
                          long m,
                          long n,
                          double *A,
                          long ldA,
                          double *B,
                          long ldB);

void launchKernel_AbsAminusAbsB(dim3 gridDim,
                                dim3 blockDim,
                                long m,
                                long n,
                                double *A,
                                long ldA,
                                double *B,
                                long ldB);



template <typename T>
void launchKernel_copyMatrix(dim3 gridDim,
                             dim3 blockDim,
                             long m,
                             long n,
                             T *srcM,
                             long lds,
                             T *dstM,
                             long ldd);


template <typename T>
void launchKernel_copyMatrixAToTranpB(dim3 gridDim,
                                      dim3 blockDim,
                                      long m,
                                      long n,
                                      T *srcM,
                                      long lds,
                                      T *dstM,
                                      long ldd);

template <typename T>
void launchKernel_getU(dim3 gridDim, dim3 blockDim, int m, int n, T *A, int ldA, T *U, int ldU);



template <typename T>
void launchKernel_getLower(dim3 gridDim, dim3 blockDim, long m, long n, T *A, long ldA);



template <typename T>
void launch_kernel_cpyATr2Vector(dim3 gridDim, dim3 blockDim, long m, long n, T *A, long ldA, T *B, cudaStream_t stream = NULL);

template <typename T>
void launch_kernel_cpyVector2ATr(dim3 gridDim, dim3 blockDim, long m, long n, T *A, long ldA, T *B, cudaStream_t stream = NULL);


void launchKernel_scaleMatrixA(dim3 gridDim,
                               dim3 blockDim,
                               long m,
                               long n,
                               double *A,
                               long ldA,
                               double scaler);

double findVectorAbsMax(double *d_array, int n);