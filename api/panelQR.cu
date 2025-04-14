#include <curand.h>
#include <cusolverDn.h>

#include <string>
#include <vector>

// #include "myBase.h"
#include "kernelOther.h"
#include "fileOpTool.h"
#include "kernelQR.h"

#include "TallShinnyQR.h"

using namespace std;

#define MY_DEBUG 0

float g_QR_Time = 0.0;
float g_Litter_GEMM_Time = 0.0;


void panelQR(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle,
             long m, long n, double *A, long lda, double *W, long ldw,
             double *R, long ldr, double *work, int *info)
{
        if (n <= 32)
        {
                startTimer();
#if MY_DEBUG
                cout << "print dA1:" << std::endl;
                string fileName = "dA1_" + to_string(m) + "_" + to_string(n) + ".csv";
                printAndWriteMatrixToCsvV2(A, lda, m, n, fileName);
#endif

                hou_tsqr_panel<128, 32>(cublas_handle, m, n, A, lda, R, ldr, work);

#if MY_DEBUG
                CHECK(cudaGetLastError());
                cudaDeviceSynchronize();



                cout << "print dQ:" << std::endl;
                fileName = "dQ_" + to_string(m) + "_" + to_string(n) + ".csv";
                printAndWriteMatrixToCsvV2(A, lda, m, n, fileName);

                // cout << "print dR:" << std::endl;
                // printDeviceMatrix(R, n, n);
                fileName = "dR_" + to_string(m) + "_" + to_string(n) + ".csv";
                printAndWriteMatrixToCsvV2(R, ldr, n, n, fileName);
#endif


                dim3 gridDim((m + 31) / 32, (n + 31) / 32);
                dim3 blockDim(32, 32);


                launchKernel_IminusQ(gridDim, blockDim, m, n, A, lda);

#if MY_DEBUG
                CHECK(cudaGetLastError());
                cudaDeviceSynchronize();
                cout << "print I-Q:" << std::endl;
                printDeviceMatrixV2(A, lda, m, n);
#endif

                launchKernel_copyMatrix(gridDim, blockDim, m, n, A, lda, W, ldw);

#if MY_DEBUG
                CHECK(cudaGetLastError());
                cudaDeviceSynchronize();

                cout << "print W(I-Q):" << std::endl;
                printDeviceMatrixV2(W, ldw, m, n);
#endif


                cusolverDnDgetrf(cusolver_handle, m, n, A, lda, work, NULL, info);
                // CHECK(cudaGetLastError());
                // cudaDeviceSynchronize();


                launchKernel_getLower(gridDim, blockDim, m, n, A, lda);
                // launchKernel_ClearMatrix(gridDim, blockDim, m, n, W, lda);



                double done = 1.0;

                cublasDtrsm(cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, m, n, &done, A, lda, W, ldw);


                g_QR_Time += stopTimer();

                return;
        }


        panelQR(cusolver_handle, cublas_handle, m, n / 2, A, lda, W, ldw, R, ldr, work, info);


        double done = 1.0, dzero = 0.0, dnegone = -1.0;

        startTimer();

        cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n / 2, n - n / 2, m,
                    &done, W, ldw, A + n / 2 * lda, lda, &dzero, work, n / 2);


        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n - n / 2, n / 2,
                    &dnegone, A, lda, work, n / 2, &done, A + n / 2 * lda, lda);
        g_Litter_GEMM_Time += stopTimer();



      
        dim3 gridDim((n / 2 + 32 - 1) / 32, (n - n / 2 + 32 - 1) / 32);
        dim3 blockDim(32, 32);

        launchKernel_copyAndClear(gridDim, blockDim, n / 2, n - n / 2, A + n / 2 * lda, lda,
                                  R + n / 2 * ldr, ldr);


        panelQR(cusolver_handle, cublas_handle, m - n / 2, n - n / 2, A + n / 2 + n / 2 * lda, lda,
                W + n / 2 + n / 2 * ldw, ldw, R + n / 2 + n / 2 * ldr, ldr, work, info);


        startTimer();
        cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n / 2, n - n / 2, m - n / 2,
                    &done, A + n / 2, lda, W + n / 2 + n / 2 * ldw, ldw, &dzero, work, n / 2);


        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n - n / 2, n / 2,
                    &dnegone, W, ldw, work, n / 2, &done, W + n / 2 * ldw, ldw);
        g_Litter_GEMM_Time += stopTimer();

        return;
}

