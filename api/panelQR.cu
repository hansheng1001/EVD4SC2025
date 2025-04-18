#include <string>
#include <vector>

#include <curand.h>
#include <cusolverDn.h>

// #include "myBase.h"
#include "fileOpTool.h"
#include "kernelOther.h"
#include "kernelQR.h"
#include "TallShinnyQR.h"

using namespace std;

#define MY_DEBUG 0

float g_QR_Time          = 0.0;
float g_Litter_GEMM_Time = 0.0;

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
             int *info)
{

  cudaDataType_t cuda_data_type;
  cublasComputeType_t cublas_compute_type;

  if (std::is_same<T, double>::value)
  {
    cuda_data_type      = CUDA_R_64F;
    cublas_compute_type = CUBLAS_COMPUTE_64F;
  }
  else if (std::is_same<T, float>::value)
  {
    cuda_data_type      = CUDA_R_32F;
    cublas_compute_type = CUBLAS_COMPUTE_32F;
  }
  else if (std::is_same<T, half>::value)
  {
    cuda_data_type      = CUDA_R_16F;
    cublas_compute_type = CUBLAS_COMPUTE_16F;
  }

  if (n <= 32)
  {
    startTimer();
#if MY_DEBUG
    cout << "print dA1:" << std::endl;
    string fileName = "dA1_" + to_string(m) + "_" + to_string(n) + ".csv";
    printAndWriteMatrixToCsvV2(A, lda, m, n, fileName);
#endif

    hou_tsqr_panel<T, 128, 32>(cublas_handle, m, n, A, lda, R, ldr, work);


    dim3 gridDim((m + 31) / 32, (n + 31) / 32);
    dim3 blockDim(32, 32);

    launchKernel_IminusQ(gridDim, blockDim, m, n, A, lda);


    launchKernel_copyMatrix(gridDim, blockDim, m, n, A, lda, W, ldw);

#if MY_DEBUG
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    cout << "print W(I-Q):" << std::endl;
    printDeviceMatrixV2(W, ldw, m, n);
#endif


    cusolverDnDgetrf(cusolver_handle, m, n, A, lda, work, NULL, info);


    launchKernel_getLower(gridDim, blockDim, m, n, A, lda);
    // launchKernel_ClearMatrix(gridDim, blockDim, m, n, W, lda);


    double done = 1.0;
    cublasDtrsm(cublas_handle,
                CUBLAS_SIDE_RIGHT,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T,
                CUBLAS_DIAG_NON_UNIT,
                m,
                n,
                &done,
                A,
                lda,
                W,
                ldw);


    g_QR_Time += stopTimer();

    return;
  }

  panelQR(cusolver_handle, cublas_handle, m, n / 2, A, lda, W, ldw, R, ldr, work, info);


  T tone    = 1.0;
  T tzero   = 0.0;
  T tnegone = -1.0;

  startTimer();


  cublasGemmEx(cublas_handle,
               CUBLAS_OP_T,
               CUBLAS_OP_N,
               n / 2,
               n - n / 2,
               m,
               &tone,
               W,
               cuda_data_type,
               ldw,

               A + n / 2 * lda,
               cuda_data_type,
               lda,

               &tzero,
               work,
               cuda_data_type,
               n / 2,
               cublas_compute_type,
               CUBLAS_GEMM_DEFAULT);


  cublasGemmEx(cublas_handle,
               CUBLAS_OP_N,
               CUBLAS_OP_N,
               m,
               n - n / 2,
               n / 2,
               &tnegone,
               A,
               cuda_data_type,
               lda,

               work,
               cuda_data_type,
               n / 2,

               &tone,
               A + n / 2 * lda,
               cuda_data_type,
               lda,
               cublas_compute_type,
               CUBLAS_GEMM_DEFAULT);


  g_Litter_GEMM_Time += stopTimer();



  dim3 gridDim((n / 2 + 32 - 1) / 32, (n - n / 2 + 32 - 1) / 32);
  dim3 blockDim(32, 32);

  launchKernel_copyAndClear(gridDim,
                            blockDim,
                            n / 2,
                            n - n / 2,
                            A + n / 2 * lda,
                            lda,
                            R + n / 2 * ldr,
                            ldr);


  panelQR(cusolver_handle,
          cublas_handle,
          m - n / 2,
          n - n / 2,
          A + n / 2 + n / 2 * lda,
          lda,
          W + n / 2 + n / 2 * ldw,
          ldw,
          R + n / 2 + n / 2 * ldr,
          ldr,
          work,
          info);


  startTimer();
  cublasGemmEx(cublas_handle,
               CUBLAS_OP_T,
               CUBLAS_OP_N,
               n / 2,
               n - n / 2,
               m - n / 2,
               &tone,
               A + n / 2,
               cuda_data_type,
               lda,

               W + n / 2 + n / 2 * ldw,
               cuda_data_type,
               ldw,

               &tzero,
               work,
               cuda_data_type,
               n / 2,
               cublas_compute_type,
               CUBLAS_GEMM_DEFAULT);


  cublasGemmEx(cublas_handle,
               CUBLAS_OP_N,
               CUBLAS_OP_N,
               m,
               n - n / 2,
               n / 2,
               &tnegone,
               W,
               cuda_data_type,
               ldw,

               work,
               cuda_data_type,
               n / 2,

               &tone,
               W + n / 2 * ldw,
               cuda_data_type,
               ldw,
               cublas_compute_type,
               CUBLAS_GEMM_DEFAULT);

  g_Litter_GEMM_Time += stopTimer();

  return;
}


template void panelQR<double>(cusolverDnHandle_t,
                              cublasHandle_t,
                              long,
                              long,
                              double *,
                              long,
                              double *,
                              long,
                              double *,
                              long,
                              double *,
                              int *);

template <>
void panelQR(cusolverDnHandle_t cusolver_handle,
             cublasHandle_t cublas_handle,
             long m,
             long n,
             float *A,
             long lda,
             float *W,
             long ldw,
             float *R,
             long ldr,
             float *work,
             int *info)
{

  cudaDataType_t cuda_data_type;
  cublasComputeType_t cublas_compute_type;

  cuda_data_type      = CUDA_R_32F;
  cublas_compute_type = CUBLAS_COMPUTE_32F;

  if (n <= 32)
  {
    startTimer();
#if MY_DEBUG
    cout << "print dA1:" << std::endl;
    string fileName = "dA1_" + to_string(m) + "_" + to_string(n) + ".csv";
    printAndWriteMatrixToCsvV2(A, lda, m, n, fileName);
#endif

    hou_tsqr_panel<float, 128, 32>(cublas_handle, m, n, A, lda, R, ldr, work);


    dim3 gridDim((m + 31) / 32, (n + 31) / 32);
    dim3 blockDim(32, 32);

    launchKernel_IminusQ(gridDim, blockDim, m, n, A, lda);


    launchKernel_copyMatrix(gridDim, blockDim, m, n, A, lda, W, ldw);

    cusolverDnSgetrf(cusolver_handle, m, n, A, lda, work, NULL, info);


    launchKernel_getLower(gridDim, blockDim, m, n, A, lda);
    // launchKernel_ClearMatrix(gridDim, blockDim, m, n, W, lda);


    float fone = 1.0;
    cublasStrsm(cublas_handle,
                CUBLAS_SIDE_RIGHT,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T,
                CUBLAS_DIAG_NON_UNIT,
                m,
                n,
                &fone,
                A,
                lda,
                W,
                ldw);


    g_QR_Time += stopTimer();

    return;
  }

  panelQR(cusolver_handle, cublas_handle, m, n / 2, A, lda, W, ldw, R, ldr, work, info);


  float tone    = 1.0;
  float tzero   = 0.0;
  float tnegone = -1.0;

  startTimer();

  cublasGemmEx(cublas_handle,
               CUBLAS_OP_T,
               CUBLAS_OP_N,
               n / 2,
               n - n / 2,
               m,
               &tone,
               W,
               cuda_data_type,
               ldw,

               A + n / 2 * lda,
               cuda_data_type,
               lda,

               &tzero,
               work,
               cuda_data_type,
               n / 2,
               cublas_compute_type,
               CUBLAS_GEMM_DEFAULT);


  cublasGemmEx(cublas_handle,
               CUBLAS_OP_N,
               CUBLAS_OP_N,
               m,
               n - n / 2,
               n / 2,
               &tnegone,
               A,
               cuda_data_type,
               lda,

               work,
               cuda_data_type,
               n / 2,

               &tone,
               A + n / 2 * lda,
               cuda_data_type,
               lda,
               cublas_compute_type,
               CUBLAS_GEMM_DEFAULT);


  g_Litter_GEMM_Time += stopTimer();



  dim3 gridDim((n / 2 + 32 - 1) / 32, (n - n / 2 + 32 - 1) / 32);
  dim3 blockDim(32, 32);

  launchKernel_copyAndClear(gridDim,
                            blockDim,
                            n / 2,
                            n - n / 2,
                            A + n / 2 * lda,
                            lda,
                            R + n / 2 * ldr,
                            ldr);

  panelQR(cusolver_handle,
          cublas_handle,
          m - n / 2,
          n - n / 2,
          A + n / 2 + n / 2 * lda,
          lda,
          W + n / 2 + n / 2 * ldw,
          ldw,
          R + n / 2 + n / 2 * ldr,
          ldr,
          work,
          info);


  startTimer();
  cublasGemmEx(cublas_handle,
               CUBLAS_OP_T,
               CUBLAS_OP_N,
               n / 2,
               n - n / 2,
               m - n / 2,
               &tone,
               A + n / 2,
               cuda_data_type,
               lda,

               W + n / 2 + n / 2 * ldw,
               cuda_data_type,
               ldw,

               &tzero,
               work,
               cuda_data_type,
               n / 2,
               cublas_compute_type,
               CUBLAS_GEMM_DEFAULT);

  cublasGemmEx(cublas_handle,
               CUBLAS_OP_N,
               CUBLAS_OP_N,
               m,
               n - n / 2,
               n / 2,
               &tnegone,
               W,
               cuda_data_type,
               ldw,

               work,
               cuda_data_type,
               n / 2,

               &tone,
               W + n / 2 *ldw,
               cuda_data_type,
               ldw,
               cublas_compute_type,
               CUBLAS_GEMM_DEFAULT);


  g_Litter_GEMM_Time += stopTimer();

  return;
}

__global__ static void
matrixCpyH2F(long int m, long int n, half *a, long int lda, float *b, long int ldb)
{
  long int i = threadIdx.x + blockDim.x * blockIdx.x;
  long int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i < m && j < n)
  {
    b[i + j * ldb] = __half2float(a[i + j * lda]);
  }
}

__global__ static void
matrixCpyF2H(long int m, long int n, float *a, long int lda, half *b, long int ldb)
{
  long int i = threadIdx.x + blockDim.x * blockIdx.x;
  long int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i < m && j < n)
  {
    b[i + j * ldb] = __float2half(a[i + j * lda]);
  }
}

template <>
void panelQR(cusolverDnHandle_t cusolver_handle,
             cublasHandle_t cublas_handle,
             long m,
             long n,
             half *A,
             long lda,
             half *W,
             long ldw,
             half *R,
             long ldr,
             half *work,
             int *info)
{

  cudaDataType_t cuda_data_type;
  cublasComputeType_t cublas_compute_type;

  cuda_data_type      = CUDA_R_16F;
  cublas_compute_type = CUBLAS_COMPUTE_16F;

  if (n <= 32)
  {
    startTimer();

    hou_tsqr_panel<half, 128, 32>(cublas_handle, m, n, A, lda, R, ldr, work);


    dim3 gridDim((m + 31) / 32, (n + 31) / 32);
    dim3 blockDim(32, 32);

    launchKernel_IminusQ(gridDim, blockDim, m, n, A, lda);


    launchKernel_copyMatrix(gridDim, blockDim, m, n, A, lda, W, ldw);


    float *_A, *_W, *_work;
    cudaMalloc((void **)&_A, sizeof(float) * m * n);
    cudaMalloc((void **)&_W, sizeof(float) * m * n);
    cudaMalloc((void **)&_work, sizeof(float) * m * n);

    matrixCpyH2F<<<gridDim, blockDim>>>(m, n, A, lda, _A, m);
    matrixCpyH2F<<<gridDim, blockDim>>>(m, n, W, ldw, _W, m);

    cusolverDnSgetrf(cusolver_handle, m, n, _A, m, _work, NULL, info);


    launchKernel_getLower(gridDim, blockDim, m, n, _A, m);


    float fone = 1.0;
    cublasStrsm(cublas_handle,
                CUBLAS_SIDE_RIGHT,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T,
                CUBLAS_DIAG_NON_UNIT,
                m,
                n,
                &fone,
                _A,
                m,
                _W,
                m);

    matrixCpyF2H<<<gridDim, blockDim>>>(m, n, _W, m, W, ldw);
    matrixCpyF2H<<<gridDim, blockDim>>>(m, n, _A, m, A, lda);

    cudaFree(_A);
    cudaFree(_W);
    cudaFree(_work);


    g_QR_Time += stopTimer();

    return;
  }

  panelQR(cusolver_handle, cublas_handle, m, n / 2, A, lda, W, ldw, R, ldr, work, info);


  float tone    = 1.0;
  float tzero   = 0.0;
  float tnegone = -1.0;

  startTimer();

  cublasGemmEx(cublas_handle,
               CUBLAS_OP_T,
               CUBLAS_OP_N,
               n / 2,
               n - n / 2,
               m,
               &tone,
               W,
               cuda_data_type,
               ldw,

               A + n / 2 * lda,
               cuda_data_type,
               lda,

               &tzero,
               work,
               cuda_data_type,
               n / 2,
               cublas_compute_type,
               CUBLAS_GEMM_DEFAULT);

 

  cublasGemmEx(cublas_handle,
               CUBLAS_OP_N,
               CUBLAS_OP_N,
               m,
               n - n / 2,
               n / 2,
               &tnegone,
               A,
               cuda_data_type,
               lda,

               work,
               cuda_data_type,
               n / 2,

               &tone,
               A + n / 2 * lda,
               cuda_data_type,
               lda,
               cublas_compute_type,
               CUBLAS_GEMM_DEFAULT);


  g_Litter_GEMM_Time += stopTimer();


  dim3 gridDim((n / 2 + 32 - 1) / 32, (n - n / 2 + 32 - 1) / 32);
  dim3 blockDim(32, 32);

  launchKernel_copyAndClear(gridDim,
                            blockDim,
                            n / 2,
                            n - n / 2,
                            A + n / 2 * lda,
                            lda,
                            R + n / 2 * ldr,
                            ldr);

  panelQR(cusolver_handle,
          cublas_handle,
          m - n / 2,
          n - n / 2,
          A + n / 2 + n / 2 * lda,
          lda,
          W + n / 2 + n / 2 * ldw,
          ldw,
          R + n / 2 + n / 2 * ldr,
          ldr,
          work,
          info);


  startTimer();
  cublasGemmEx(cublas_handle,
               CUBLAS_OP_T,
               CUBLAS_OP_N,
               n / 2,
               n - n / 2,
               m - n / 2,
               &tone,
               A + n / 2,
               cuda_data_type,
               lda,

               W + n / 2 + n / 2 * ldw,
               cuda_data_type,
               ldw,

               &tzero,
               work,
               cuda_data_type,
               n / 2,
               cublas_compute_type,
               CUBLAS_GEMM_DEFAULT);


  cublasGemmEx(cublas_handle,
               CUBLAS_OP_N,
               CUBLAS_OP_N,
               m,
               n - n / 2,
               n / 2,
               &tnegone,
               W,
               cuda_data_type,
               ldw,

               work,
               cuda_data_type,
               n / 2,

               &tone,
               W + n / 2 * ldw,
               cuda_data_type,
               ldw,
               cublas_compute_type,
               CUBLAS_GEMM_DEFAULT);


  g_Litter_GEMM_Time += stopTimer();

  return;
}
