#include <string>
#include <vector>

#include <iostream>
#include <assert.h>

#include "PanelQR.h"
#include "kernelOther.h"
#include "fileOpTool.h"


#include "TallShinnyQR.h"

#include "myBase.h"

#include "tc_ozimmu_syr2k.h"

using namespace std;

float g_panelQR_time_ZY = 0.0;
float g_tc_ozimmu_syr2k_ZY = 0.0;

float g_gemm_time_ZY = 0.0;


// #define MY_DEBUG 1

void my_ZY_ZY_SBR(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle,
                  long M, long N, long b, long nb, double *dOriA, long ldOriA, double *dA, long ldA,
                  double *dW, long ldW, double *dY, long ldY, double *dZ, long ldZ, double *dR, long ldR, double *work, int *info)
{

        if (0 >= M)
        {
                return;
        }


        if (0 != (M % nb))
        {
                cout << "M must be diviable by nb!" << endl;
                return;
        }

        // double *dwork;
        // cudaMalloc(&dwork, sizeof(double) * (M * nb));

        double done = 1.0;
        double dzero = 0.0;
        double dnegone = -1.0;
        double dneghalf = -0.5;


        double *OA = dOriA + b * ldOriA + b;

        long ldWork = ldOriA;

        long i;
        for (i = b; (i <= nb) && (i < N); i += b)
        {

                long m = M - i;
                long n = b;


                double *dPanel = dA + i + (i - b) * ldA;


                double *dPanelW = dW + i + (i - b) * ldW;


                double *dPanelR = dR + i + (i - b) * ldR;

#if MY_DEBUG
                cout << "print dPanelA:" << endl;
                printDeviceMatrixV2(dPanel, ldA, 32, 32);
#endif


                startTimer();
                panelQR(cusolver_handle, cublas_handle, m, n, dPanel, ldA, dPanelW, ldW, dPanelR, ldR, work, info);
                // panelQR(cusolver_handle, cublas_handle, m, n, dA + i + (i - b) * ldA, ldA,
                //         dW + i + (i - b) * ldW, ldW, dR + i + (i - b) * ldR, ldR, work, info);
                g_panelQR_time_ZY += stopTimer();

                CHECK(cudaGetLastError());
                cudaDeviceSynchronize();


                dim3 gridDim((m + 31) / 32, (n + 31) / 32);
                dim3 blockDim(32, 32);


                double *dPanelY = dY + i + (i - b) * ldY;
                launchKernel_copyMatrix(gridDim, blockDim, m, n, dPanel, ldA, dPanelY, ldY);

                launchKernel_getU(gridDim, blockDim, m, n, dPanelR, ldR, dPanel, ldA);




                double *dPanelZ = dZ + i + (i - b) * ldZ;

                startTimer();
                if (i == b)
                {

                        cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, b, m,
                                     &done, OA, CUDA_R_64F, ldOriA, dPanelW, CUDA_R_64F, ldW, &dzero, dPanelZ,
                                     CUDA_R_64F, ldZ, CUDA_R_64F, CUBLAS_GEMM_DEFAULT);


                        cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, b, b, m,
                                     &done, dPanelW, CUDA_R_64F, ldW, dPanelZ, CUDA_R_64F, ldZ, &dzero, work,
                                     CUDA_R_64F, ldWork, CUDA_R_64F, CUBLAS_GEMM_DEFAULT);


                        cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, b, b,
                                     &dneghalf, dPanelY, CUDA_R_64F, ldY, work, CUDA_R_64F, ldWork, &done, dPanelZ,
                                     CUDA_R_64F, ldZ, CUDA_R_64F, CUBLAS_GEMM_DEFAULT);
                }
                else
                {

                        cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, b, m,
                                     &done, OA + (i - b) + (i - b) * ldOriA, CUDA_R_64F, ldOriA,
                                     dPanelW, CUDA_R_64F, ldW, &dzero, dPanelZ,
                                     CUDA_R_64F, ldZ, CUDA_R_64F, CUBLAS_GEMM_DEFAULT);


                        cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, i - b, b, m,
                                     &done, dZ + i, CUDA_R_64F, ldZ,
                                     dPanelW, CUDA_R_64F, ldW, &dzero, work,
                                     CUDA_R_64F, ldWork, CUDA_R_64F, CUBLAS_GEMM_DEFAULT);

                        cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, b, i - b,
                                     &dnegone, dY + i, CUDA_R_64F, ldY,
                                     work, CUDA_R_64F, ldWork, &done, dPanelZ,
                                     CUDA_R_64F, ldZ, CUDA_R_64F, CUBLAS_GEMM_DEFAULT);


                        cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, i - b, b, m,
                                     &done, dY + i, CUDA_R_64F, ldY,
                                     dPanelW, CUDA_R_64F, ldW, &dzero, work,
                                     CUDA_R_64F, ldWork, CUDA_R_64F, CUBLAS_GEMM_DEFAULT);


                        cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, b, i - b,
                                     &dnegone, dZ + i, CUDA_R_64F, ldZ,
                                     work, CUDA_R_64F, ldWork, &done, dPanelZ,
                                     CUDA_R_64F, ldZ, CUDA_R_64F, CUBLAS_GEMM_DEFAULT);


                        cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, b, b, m,
                                     &done, dPanelW, CUDA_R_64F, ldW,
                                     dPanelZ, CUDA_R_64F, ldZ, &dzero, work,
                                     CUDA_R_64F, ldWork, CUDA_R_64F, CUBLAS_GEMM_DEFAULT);

                        cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, b, b,
                                     &dneghalf, dPanelY, CUDA_R_64F, ldY, work, CUDA_R_64F, ldWork, &done, dPanelZ,
                                     CUDA_R_64F, ldZ, CUDA_R_64F, CUBLAS_GEMM_DEFAULT);
                }


                if (i < nb)
                {


                        cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m, b, i,
                                     &dnegone, dY + i, CUDA_R_64F, ldY, dZ + i, CUDA_R_64F, ldZ, &done, dA + i + i * ldA,
                                     CUDA_R_64F, ldA, CUDA_R_64F, CUBLAS_GEMM_DEFAULT);


                        cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m, b, i,
                                     &dnegone, dZ + i, CUDA_R_64F, ldZ, dY + i, CUDA_R_64F, ldY, &done, dA + i + i * ldA,
                                     CUDA_R_64F, ldA, CUDA_R_64F, CUBLAS_GEMM_DEFAULT);
                }

                g_gemm_time_ZY += stopTimer();

#if MY_DEBUG
                cout << "print dGA:" << endl;
                printDeviceMatrixV2(dA + b + i * ldA, ldA, M - b, b);
#endif
        }



        if (0 >= N - nb)
        {
#if MY_DEBUG
                cout << "SBR end!" << endl;
#endif
                return;
        }



        long lm = M - nb; 
        long ln = nb;

        dim3 block1(32, 32);
        dim3 grid1((lm + 31) / 32, (lm + 31) / 32);

        startTimer();
        if (lm > 32768)
        {
                tc_ozimmu_syr2k(cublas_handle, lm, ln,
                                dnegone, dY + nb, ldY,
                                dZ + nb, ldZ, done,
                                OA + (nb - b) + (nb - b) * ldOriA, ldOriA, nb);
                // OA, ldOriA, nb);
        }
        else
        {
                cublasDsyr2k(cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, lm, ln, &dnegone, dY + nb, ldY,
                             dZ + nb, ldZ, &done, OA + (nb - b) + (nb - b) * ldOriA, ldOriA);
        }
        g_tc_ozimmu_syr2k_ZY += stopTimer();

        launchKernel_CpyMatrixL2U(grid1, block1, lm, OA + (nb - b) + (nb - b) * ldOriA, ldOriA);

        // printf("OA:\n");
        // printDeviceMatrixV2(OA, ldOriA, 10, 10);

        // printf("OA end 10:\n");
        // printDeviceMatrixV2(OA + (lm - 10) + (lm - 10) * ldOriA, ldOriA, 10, 10);

        dim3 gridDim2((M - nb + 31) / 32, (N - nb + 31) / 32);
        dim3 blockDim2(32, 32);

        // cudaFree(dwork);


        launchKernel_copyMatrix(gridDim2, blockDim2, M - nb, N - nb,
                                dOriA + nb + nb * ldOriA, ldOriA, dA + nb + nb * ldA, ldA);

        M = N = M - nb;
        dA = dA + nb + nb * ldA;


        lm = M - b;
        dim3 grid2((lm + 31) / 32, (ln + 31) / 32);

        launchKernel_ClearMatrix(grid2, block1, lm, ln, dW + b, ldW);
        launchKernel_ClearMatrix(grid2, block1, lm, ln, dY + b, ldY);
        launchKernel_ClearMatrix(grid2, block1, lm, ln, dZ + b, ldZ);

        dW = dW + nb;
        dY = dY + nb;
        dR = dR + nb;

        dOriA = dOriA + nb + nb * ldOriA;

        my_ZY_ZY_SBR(cusolver_handle, cublas_handle, M, N, b, nb, dOriA, ldOriA, dA, ldA, dW, ldW, dY, ldY, dZ, ldZ, dR, ldR, work, info);
}


