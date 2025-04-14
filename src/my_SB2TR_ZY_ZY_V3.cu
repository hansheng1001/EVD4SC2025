
#include <string>
#include <vector>

#include <iostream>
#include <iomanip> 

// #include "bugle_chasing.h"
// #include "bugle_chasingV3.h"
#include "bugle_chasingV7.h"
#include "kernelOther.h"
#include "fileOpTool.h"
#include "myBase.h"
#include "zy_zy_sy2sb.h"

using namespace std;

float g_sy2sb_time = 0.0;
float g_bugle_chasing_kernel_time = 0.0;

float g_cusolverSy2tr_Time = 0.0;

extern float g_panelQR_time_ZY;
extern float g_tc_ozimmu_syr2k_ZY;
extern float g_gemm_time_ZY;

#define CUSOLVER_CHECK 0
#define MY_SY2SB 1


static __global__ void kernel_bugle_chasing_cpydA2dSubA(int n, int b, double *dA, long ldA,
                                                        double *dSubA, int ldSubA)
// static __global__ void kernel_bugle_chasing_cpydA2dSubA(int n, int b, double *dSubA, int ldSubA,
//                                                         double *dA, int ldA)
{

    // int bInx = blockIdx.y * gridDim.x + blockIdx.x;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;



    // if (i < (b + 1) && j < n)
    if ((i < 2 * b) && j < n)
    {
 
        int end = min(n, j + b + 1); 

   
        int count = end - j;

        // printf("block[%d] [%d][%d] come line=%d,count=%d.\n", bInx, i, j, __LINE__, count);

        if (i < count)
        {
            dSubA[i + j * ldSubA] = dA[j + i + j * ldA];
        }
        else
        {
            dSubA[i + j * ldSubA] = 0.0;
        }
    }
}

static __global__ void kernel_bugle_chasing_cpydA2dSubA_V2(int n, int b, double *dA, int ldA,
                                                           double *dSubA, int ldSubA)
{

    // int bInx = blockIdx.y * gridDim.x + blockIdx.x;

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // if (i < (b + 1) && j < n)
    if (i < n)
    {

        // int end = min(n, j + 2b);
        int end = min(n, i + b + 1); 

        
        int count = end - i;

        // printf("block[%d] [%d][%d] come line=%d,count=%d.\n", bInx, i, j, __LINE__, count);

        for (int k = 0; k < count; k++)
        {
            dSubA[k + i * ldSubA] = dA[i + k + i * ldA];
        }

        for (int k = count; k < 2 * b; k++)
        {
            dSubA[k + i * ldSubA] = 0.0;
        }
    }
}


static __global__ void kernel_bugle_chasing_cpydSubA2dA(int n, int b, double *dSubA, int ldSubA,
                                                        double *dA, long ldA)
{
    // int i = bInx * blockDim.x + threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;


    if (i < n)
    {
        dA[i + i * ldA] = dSubA[i * ldSubA];

        if (i < n - 1)
        {
            dA[i + 1 + i * ldA] = dSubA[i * ldSubA + 1];
        }
    }
}


int main(int argc, char *argv[])
{

    long m, n;
    long b = 32;

    long nb = 4 * b;





    if (4 != argc)
    {
        cout << "Usage(b = nb in ZY): AppName <n> <b> <nb>" << endl;
        return 0;
    }

    m = n = atol(argv[1]);
    b = atol(argv[2]);
    nb = atol(argv[3]);

    // m = n = 16384;
    // b = 32;
    // nb = 1024;

    cout << "My Sy2tr use ZY_ZY V2:" << endl;
    cout << "n=" << n << ", b=" << b << ", nb=" << nb << endl;



    // cudaSetDevice(4);

    cusolverDnHandle_t cusolver_handle;
    cublasHandle_t cublas_handle;

    cusolverDnCreate(&cusolver_handle);
    cublasCreate(&cublas_handle);



    double *dT, *dA;
    cudaMalloc(&dT, sizeof(double) * m * n);
    cudaMalloc(&dA, sizeof(double) * m * n);
    // cudaMalloc(&dWork, sizeof(double) * m * n);

    // cudaMemcpy(dT, A, sizeof(double) * m * n, cudaMemcpyHostToDevice);


    generateUniformMatrix(dT, m, n);


    dim3 gridDim((m + 31) / 32, (n + 31) / 32);
    dim3 blockDim(32, 32);
    launchKernel_CpyMatrixL2U(gridDim, blockDim, n, dT, n);

    launchKernel_copyMatrix(gridDim, blockDim, m, n, dT, m, dA, m);





    cudaFree(dT);

    long ldA = m;

#if MY_SY2SB
    double *dwork, *dR, *dW, *dY, *dZ;

    // cudaMalloc(&dwork, sizeof(double) * (m + nb) * (n + nb));
    // cudaMalloc(&dR, sizeof(double) * m * n);
    // cudaMalloc(&dW, sizeof(double) * m * n);
    // cudaMalloc(&dY, sizeof(double) * m * n);
    cudaMalloc(&dwork, sizeof(double) * m * nb);
    cudaMalloc(&dR, sizeof(double) * m * nb);
    cudaMalloc(&dW, sizeof(double) * m * nb);
    cudaMalloc(&dY, sizeof(double) * m * nb);

    cudaMalloc(&dZ, sizeof(double) * m * nb);





    int *info;
    cudaMalloc(&info, sizeof(int));

    CHECK(cudaGetLastError());

    double *dOriA_1;
    cudaMalloc(&dOriA_1, sizeof(double) * m * n);

    // cudaMemcpy(dOriA_1, dA, sizeof(double) * m * n, cudaMemcpyDeviceToDevice);
    // cudaMemcpy(dOriA_1, A, sizeof(double) * m * n, cudaMemcpyHostToDevice);

    launchKernel_copyMatrix(gridDim, blockDim, m, n, dA, m, dOriA_1, m);

    long ldOriA_1, ldW, ldY, ldZ, ldR;

    ldOriA_1 = ldW = ldY = ldZ = ldR = m;



    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    startTimer();
    my_ZY_ZY_SBR(cusolver_handle, cublas_handle, m, n, b, nb, dOriA_1, ldOriA_1,
                 dA, ldA, dW, ldW, dY, ldY, dZ, ldZ, dR, ldR, dwork, info);

    launchKernel_CpyMatrixL2U(gridDim, blockDim, m, dA, ldA);
    g_sy2sb_time = stopTimer();

    printf("SBR dA:\n");

    printDeviceMatrixV2(dA, ldA, 3, 3);


    printDeviceMatrixV2(dA + (m - 3) + (n - 3) * ldA, ldA, 3, 3);



    // printDeviceMatrixV2(dA, ldA, m, n);

    cudaFree(dR);
    cudaFree(dW);
    cudaFree(dY);
    cudaFree(dZ);

    cudaFree(dOriA_1);
    cudaFree(dwork);
#endif

    double *dSubA;
    cudaMalloc(&dSubA, sizeof(double) * (2 * b) * n);
    int ldSubA = 2 * b;




    // dim3 blockDimBugleChasing(32, 32);
    dim3 blockDimBugleChasing(32, 32);

    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    

    startTimer();


    dim3 blockDimcpydA2dSubA(32, 32);
    // dim3 gridDimcpydA2dSubA((b + 1 + 31) / 32, (n + 31) / 32);
    dim3 gridDimcpydA2dSubA((2 * b + 31) / 32, (n + 31) / 32);
    kernel_bugle_chasing_cpydA2dSubA<<<gridDimcpydA2dSubA, blockDimcpydA2dSubA>>>(n, b, dA, ldA, dSubA, ldSubA);

 

    // my_bugle_chasing_kernelV3<32, 32><<<n - 2, blockDimBugleChasing>>>(n, b, dA, ldA)
        chasing_kernel_one_timeV7<32><<<n - 2, blockDimBugleChasing>>>(n, b, dSubA, ldSubA);

    // CHECK(cudaGetLastError());
    // cudaDeviceSynchronize();

    // g_bugle_chasing_kernel_time = stopTimer();


    dim3 blockDimcpydSubA2dA(32);
    dim3 gridDimcpydSubA2dA((n + 31) / 32);

    kernel_bugle_chasing_cpydSubA2dA<<<gridDimcpydSubA2dA, blockDimcpydSubA2dA>>>(n, b, dSubA, ldSubA, dA, ldA);

    g_bugle_chasing_kernel_time = stopTimer();

    // my_hou_kernel<128, 32><<<gridDim, blockDim>>>(96, 32, dA + 32, m, dR, n);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    printf("SY2TR dA:\n");

    printDeviceMatrixV2(dA, ldA, 3, 3);


    printDeviceMatrixV2(dA + (m - 3) + (n - 3) * ldA, ldA, 3, 3);



    printf("gemm %ldx%ld takes %lf ms, tflops is %lf\n", m, n, g_gemm_time_ZY,
           2.0 * n * n * (m - 1.0 / 3.0 * n) / (g_gemm_time_ZY * 1e9));

    printf("syr2k %ldx%ld takes %lf ms, tflops is %lf\n", m, n, g_tc_ozimmu_syr2k_ZY,
           2.0 * n * n * (m - 1.0 / 3.0 * n) / (g_tc_ozimmu_syr2k_ZY * 1e9));

    printf("qr %ldx%ld takes %lf ms, tflops is %lf\n", m, n, g_panelQR_time_ZY,
           2.0 * n * n * (m - 1.0 / 3.0 * n) / (g_panelQR_time_ZY * 1e9));

    printf("sy2sb %ldx%ld takes %lf ms, tflops is %lf\n", m, n, g_sy2sb_time,
           2.0 * n * n * (m - 1.0 / 3.0 * n) / (g_sy2sb_time * 1e9));

    printf("Bugle chasing %ldx%ld takes %lf ms, tflops is %lf\n", m, n, g_bugle_chasing_kernel_time,
           2.0 * n * n * (m - 1.0 / 3.0 * n) / (g_bugle_chasing_kernel_time * 1e9));

    float ms = g_sy2sb_time + g_bugle_chasing_kernel_time;
    printf("sy2tr %ldx%ld takes %lf ms, tflops is %lf\n", m, n,
           ms, (4.0 * n * n * n / 3.0) / (ms * 1e9));
}
