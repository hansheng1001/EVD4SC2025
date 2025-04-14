#pragma once

#include <cusolverDn.h>
#include <iostream>
#include "kernelQR.h"
#include "myBase.h"


template <long M, long N>
void hou_tsqr_panel(cublasHandle_t cublas_handle, long m, long n, double *A, long lda, double *R, long ldr, double *work)
{
    if (n > N)
    {
        std::cout << "hou_tsqr_panel QR the n must <= N" << std::endl;
        exit(1);
    }

    dim3 blockDim(32, 16);

    if (m <= M)
    {

        my_hou_kernel<M, N><<<1, blockDim>>>(m, n, A, lda, R, ldr);
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        return;
    }

    if (0 != (m % M) % n)
    {
        std::cout << "hou_tsqr_panel QR the m%M must be multis of n" << std::endl;
        exit(1);
    }


    long blockNum = (m + M - 1) / M;
    long ldwork = blockNum * n;


    my_hou_kernel<M, N><<<blockNum, blockDim>>>(m, n, A, lda, work, ldwork);
    // std::cout << __func__ << " " << __LINE__ << " m=" << m << ", n=" << n << ", M=" << M << std::endl;
    // CHECK(cudaGetLastError());
    // cudaDeviceSynchronize();

    hou_tsqr_panel<M, N>(cublas_handle, ldwork, n, work, ldwork, R, ldr, work + n * ldwork);

    // std::cout << "print dR:" << std::endl;

    double done = 1.0, dzero = 0.0;
    cublasDgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, n, n, &done, A, lda, M, work, ldwork, n, &dzero, A, lda, M, m / M);

    // std::cout << __func__ << " " << __LINE__ << " m=" << m << ", n=" << n << ", M=" << M << std::endl;

    // CHECK(cudaGetLastError());
    // cudaDeviceSynchronize();

    // std::cout << "print dA:" << std::endl;
    // printDeviceMatrixV2(A, lda, m, n);


    long mm = m % M;
    if (0 < mm)
    {
#if MY_DEBUG
        std::cout << __func__ << " " << __LINE__ << " come m % M !=0 case." << std::endl;
#endif

        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, mm, n, n, &done, A + (m - mm), lda, work + (m / M * n), ldwork, &dzero, A + (m - mm), lda);
    }

    // std::cout << "print dA:" << std::endl;
    // printDeviceMatrixV2(A, lda, m, n);
}