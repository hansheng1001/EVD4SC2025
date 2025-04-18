
#include "computerQFromWY.h"
#include "kernelOther.h"

void computerQFromWY(cublasHandle_t cublas_handle, long M, long N, double *dQ, long ldQ,
                     double *dW, long ldW, double *dY, long ldY)
{
    double done = 1.0;
    double dzero = 0.0;

    cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, M, M, N,
                &done, dW, ldW, dY, ldY, &dzero, dQ, ldQ);


    dim3 gridDim((M + 31) / 32, (M + 31) / 32);
    dim3 blockDim(32, 32);
    launchKernel_IminusQ(gridDim, blockDim, M, M, dQ, ldQ);
}