
#include <cstring>
#include <iomanip> 
#include <iostream>
#include <string>
#include <vector>
// #include <lapacke.h>
#include <mkl_lapacke.h>
#include <algorithm> // std::sort

#include "BC_backTrans.h"
#include "bugle_chasingV10.h"
#include "fileOpTool.h"
#include "kernelOther.h"
#include "myBase.h"
#include "zy_zy_sy2sb.h"

#include "computerWYFromSlide.h"
#include "computerQFromWY.h"
#include "checkMetric.h"

#include <thread>


#define TESTING_CHECK( err )                                                 \
    do {                                                                     \
        magma_int_t err_ = (err);                                            \
        if ( err_ != 0 ) {                                                   \
            fprintf( stderr, "Error: %s\nfailed at %s:%d: error %lld: %s\n", \
                     #err, __FILE__, __LINE__,                               \
                     (long long) err_, magma_strerror(err_) );               \
            exit(1);                                                         \
        }                                                                    \
    } while( 0 )



using namespace std;

float g_sy2sb_time                = 0.0;
float g_bugle_chasing_kernel_time = 0.0;


float g_cusolverSy2tr_Time = 0.0;

extern float g_panelQR_time_ZY;
extern float g_tc_ozimmu_syr2k_ZY;
extern float g_gemm_time_ZY;

#define CUSOLVER_CHECK 0


static __global__ void
kernel_bugle_chasing_cpydA2dSubA(int n, int b, double *dA, long ldA, double *dSubA, int ldSubA)
{


  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;


  if ((i < 2 * b) && j < n)
  {
    int end = min(n, j + b + 1); 

    int count = end - j;


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

static __global__ void
kernel_bugle_chasing_cpydA2dSubA_V2(int n, int b, double *dA, int ldA, double *dSubA, int ldSubA)
{

  int i = blockIdx.x * blockDim.x + threadIdx.x;


  if (i < n)
  {
    int end = min(n, i + b + 1); 

    int count = end - i;

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

static __global__ void
kernel_bugle_chasing_cpydSubA2dA(int n, int b, double *dSubA, int ldSubA, double *dA, long ldA)
{
  // int i = bInx * blockDim.x + threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

 
  if (i < n)
  {
    dA[i + i * ldA] = dSubA[i * ldSubA];

    if (i < n - 1)
    {
      dA[i + 1 + i * ldA] = dSubA[i * ldSubA + 1];

      dA[i + (i + 1) * ldA] = dSubA[i * ldSubA + 1];
    }
  }
}

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "CUBLAS error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

void my_SBRBack_BCBack(cublasHandle_t cublas_handle, double * dQ, long ldQ, double *dU, long ldU, 
            double * dW, long ldW, double * dY, long ldY, double *dwork,
            long n, long b, cudaStream_t stream = NULL)
{
  double done = 1.0;
  double dzero = 0.0;
  double dnegone = -1.0;

  long nk  = 1024;

  auto start1 = std::chrono::high_resolution_clock::now();
  
  // CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
  for (long col_Wk = b; col_Wk < nk; col_Wk *= 2)
  {

      cublasGemmStridedBatchedEx(cublas_handle,
                                  CUBLAS_OP_T,
                                  CUBLAS_OP_N,
                                  col_Wk,
                                  col_Wk,
                                  n - b,

                                  &done,
                                  dY + b,
                                  CUDA_R_64F,
                                  ldY,
                                  2 * col_Wk * ldY,

                                  dW + b + col_Wk * ldW,
                                  CUDA_R_64F,
                                  ldW,
                                  2 * col_Wk * ldW,

                                  &dzero,
                                  dwork,
                                  CUDA_R_64F,
                                  n,
                                  2 * col_Wk * n,

                                  n / (2 * col_Wk),
                                  CUBLAS_COMPUTE_64F,
                                  CUBLAS_GEMM_DEFAULT);


      cublasGemmStridedBatchedEx(cublas_handle,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  n - b,
                                  col_Wk,
                                  col_Wk,

                                  &dnegone,
                                  dW + b,
                                  CUDA_R_64F,
                                  ldW,
                                  2 * col_Wk * ldW,

                                  dwork,
                                  CUDA_R_64F,
                                  n,
                                  2 * col_Wk * n,

                                  &done,
                                  dW + b + col_Wk * ldW,
                                  CUDA_R_64F,
                                  ldW,
                                  2 * col_Wk * ldW,

                                  n / (2 * col_Wk),
                                  CUBLAS_COMPUTE_64F,
                                  CUBLAS_GEMM_DEFAULT);
  }

  int count = n / nk;



  for (int i = count - 1; i >= 0; i--)
  {

      cublasDtrmm(cublas_handle,
                  CUBLAS_SIDE_LEFT,
                  CUBLAS_FILL_MODE_LOWER,
                  CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                  nk, n,
                  &done,
                  dY + (i * nk * ldY), ldY,
                  dQ, ldQ,
                  dwork, n);


      cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, nk, n, n - nk, &done, dY + nk + (i * nk * ldY), ldY,
                  dQ + nk, ldQ, &done, dwork, n);


      cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, nk, &dnegone, dW + (i * nk * ldW), ldW,
                  dwork, n, &done, dQ, ldQ);
  }

  dim3 gridDim((n + 31) / 32, (n + 31) / 32);
  dim3 blockDim(32, 32);

  launchKernel_copyMatrixAToTranpB(gridDim, blockDim, n, n, dQ, ldQ, dwork, n);
  launchKernel_copyMatrix(gridDim, blockDim, n, n, dwork, n, dQ, ldQ);

  cudaStreamSynchronize(stream);
  auto end1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration1 = end1 - start1;

  std::cout << "SBR Back time: " << duration1.count() << " ms" << std::endl;


  my_BC_back_trans_v8_10_noBandU(dQ, ldQ, dU, ldU, n, b);
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
  b     = atol(argv[2]);
  nb    = atol(argv[3]);

  cout << "My Sy2tr use ZY_ZY V5:" << endl;
  cout << "n=" << n << ", b=" << b << ", nb=" << nb << endl;


  cusolverDnHandle_t cusolver_handle;
  cublasHandle_t cublas_handle;

  cusolverDnCreate(&cusolver_handle);
  cublasCreate(&cublas_handle);


  double *dT, *dA;
  cudaMalloc(&dT, sizeof(double) * m * n);
  cudaMalloc(&dA, sizeof(double) * m * n);

  generateUniformMatrix(dT, m, n);

  dim3 gridDim((m + 31) / 32, (n + 31) / 32);
  dim3 blockDim(32, 32);
  launchKernel_CpyMatrixL2U(gridDim, blockDim, n, dT, n);

  launchKernel_copyMatrix(gridDim, blockDim, m, n, dT, m, dA, m);

#define CHECH_EVD_RESULT_ENABLE 0
#if !(CHECH_EVD_RESULT_ENABLE) 
  cudaFree(dT);
#endif


  double *dwork, *dR, *dW, *dY, *dZ;

  cudaMalloc(&dwork, sizeof(double) * (m + nb) * (n + nb));
  cudaMalloc(&dW, sizeof(double) * m * n);
  cudaMalloc(&dY, sizeof(double) * m * n);
  cudaMalloc(&dR, sizeof(double) * m * nb);

  cudaMalloc(&dZ, sizeof(double) * m * nb);

  int *info;
  cudaMalloc(&info, sizeof(int));

  CHECK(cudaGetLastError());

  double *dOriA_1;
  cudaMalloc(&dOriA_1, sizeof(double) * m * n);

  launchKernel_copyMatrix(gridDim, blockDim, m, n, dA, m, dOriA_1, m);

  long ldOriA_1, ldA, ldW, ldY, ldZ, ldR;
  ldA      = m;
  ldOriA_1 = ldW = ldY = ldZ = ldR = m;


  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  startTimer();
  my_ZY_ZY_SBR_Vector(cusolver_handle,
               cublas_handle,
               m,
               n,
               b,
               nb,
               dOriA_1,
               ldOriA_1,
               dA,
               ldA,
               dW,
               ldW,
               dY,
               ldY,
               dZ,
               ldZ,
               dR,
               ldR,
               dwork,
               info);

  launchKernel_CpyMatrixL2U(gridDim, blockDim, m, dA, ldA);
  g_sy2sb_time = stopTimer();

  printf("SBR dA:\n");
  printDeviceMatrixV2(dA, ldA, 3, 3);

  printDeviceMatrixV2(dA + (m - 3) + (n - 3) * ldA, ldA, 3, 3);


  cudaFree(dR);
  cudaFree(dZ);

  cudaFree(dOriA_1);


  double *dSubA;
  cudaMalloc(&dSubA, sizeof(double) * (2 * b) * n);
  int ldSubA = 2 * b;

  dim3 blockDimBugleChasing(32, 32);

  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  int dev                = 0;

  int supportsCoopLaunch = 0;
  cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);

  printf("Device %d supports cooperative launch: %s\n", dev, supportsCoopLaunch ? "true" : "false");

  int numBlocksPerSm = 0;
  int numThreads = 32 * 32;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);


  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm,
      chasing_kernel_one_timeV10<32>,
      numThreads,
      0);
  int blockNum = numBlocksPerSm * deviceProp.multiProcessorCount;

  printf("maxThreadsPerBlock: %d\n", deviceProp.maxThreadsPerBlock);
  printf("multiProcessorCount: %d\n", deviceProp.multiProcessorCount);
  printf("numBlocksPerSm: %d\n", numBlocksPerSm);

  printf("blockNum: %d\n", blockNum);

  startTimer();

  dim3 blockDimcpydA2dSubA(32, 32);
  dim3 gridDimcpydA2dSubA((2 * b + 31) / 32, (n + 31) / 32);
  kernel_bugle_chasing_cpydA2dSubA<<<gridDimcpydA2dSubA, blockDimcpydA2dSubA>>>(n,
                                                                                b,
                                                                                dA,
                                                                                ldA,
                                                                                dSubA,
                                                                                ldSubA);

  double *dU;
  cudaMalloc(&dU, sizeof(double) * (m+nb) * n);
  long ldU = m + nb;

  dim3 blockDimClrDUOnly(32, 32);
  dim3 gridDimClrDuOnly((m+nb + 31) / 32, (n + 31) / 32);
  launchKernel_ClearMatrix(gridDimClrDuOnly, blockDimClrDUOnly, m, n, dU, ldU);

  dim3 blockDimClrDU(32, 32);
  dim3 gridDimClrDu((m + 31) / 32, (n + 31) / 32);
  launchKernel_ClearMatrix(gridDimClrDu, blockDimClrDU, m, n, dA, ldA);

  int *com;
  cudaMalloc(&com, n*sizeof(int));


  void *kernelArgs[] = {
      (void *)&n,
      (void *)&b,
      (void *)&dSubA,
      (void *)&ldSubA,
      (void *)&dU,
      (void *)&ldU,
      (void *)&blockNum,
      (void *)&com,
      // (void *)&g_overFlag,
  };
  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid(blockNum, 1, 1);
  cudaLaunchCooperativeKernel((void *)chasing_kernel_one_timeV10<32>,
                              dimGrid,
                              dimBlock,
                              kernelArgs);

  dim3 blockDimcpydSubA2dA(32);
  dim3 gridDimcpydSubA2dA((n + 31) / 32);

  kernel_bugle_chasing_cpydSubA2dA<<<gridDimcpydSubA2dA, blockDimcpydSubA2dA>>>(n,
                                                                                b,
                                                                                dSubA,
                                                                                ldSubA,
                                                                                dA,
                                                                                ldA);

  g_bugle_chasing_kernel_time = stopTimer();

  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  cudaFree(dSubA);
  cudaFree(com);

  printf("SY2TR dA:\n");
  printDeviceMatrixV2(dA, ldA, 3, 3);

  printDeviceMatrixV2(dA + (m - 3) + (n - 3) * ldA, ldA, 3, 3);


#define USE_LAPACKE 1
#if USE_LAPACKE
  double *mD, *mE;
  cudaMalloc(&mD, sizeof(double) *n);
  cudaMalloc(&mE, sizeof(double) *(n - 1));

  double *hS, *hE;
  hS  = (double *)malloc(sizeof(double) * n);
  hE  = (double *)malloc(sizeof(double) * (n - 1));

  dim3 blockDim4(32);
  dim3 gridDim4((m + 31) / 32);
  launch_kernel_cpyATr2Vector(gridDim4, blockDim4, n, n, dA, ldA, mD);

  dim3 gridDim5((m - 1 + 31) / 32);
  launch_kernel_cpyATr2Vector(gridDim5, blockDim4, n - 1, n - 1, dA + 1, ldA, mE);

  cudaMemcpy(hS, mD, sizeof(double) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(hE, mE, sizeof(double) * (n - 1), cudaMemcpyDeviceToHost);

  double *Z;
  Z = (double *)malloc(sizeof(double) * n*n); 

  float lapackestedcTime;
#endif

  cudaFree(dA);

#define BUGLE_CHASING_COMPUTER_Q 1
#if BUGLE_CHASING_COMPUTER_Q

  CUDA_RT_CALL(cudaGetLastError());
  cudaDeviceSynchronize();

  double *dQ;
  cudaMalloc(&dQ, sizeof(double) * (m+nb) * n);
  long ldQ = m + nb;

  launchKernel_ClearMatrix(gridDimClrDu, blockDimClrDU, m, n, dQ, ldQ);
  double value = 1.0;
  dim3 blockDimSetValue(32);
  dim3 gridDimSetValue((n + 31) / 32);
  launchKernel_setMetrixTrValue(gridDimSetValue, blockDimSetValue, n, n, dQ, ldQ, value);

  CUDA_RT_CALL(cudaGetLastError());
  cudaDeviceSynchronize();


  float BC_Back_time;

  auto start = std::chrono::high_resolution_clock::now();


  std::thread cpuThread(my_SBRBack_BCBack,cublas_handle, dQ, ldQ, dU, ldU, dW, ldW, dY, ldY, dwork, n, b, (cudaStream_t) NULL);
  
#endif

#if USE_LAPACKE

  {
    auto start1 = std::chrono::high_resolution_clock::now();

    lapack_int lapckInfo=0;

    lapckInfo = LAPACKE_dstedc(LAPACK_COL_MAJOR, 'I', n, hS, hE, Z, n);

    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration1 = end1 - start1;

    std::cout << "MKL DC time: " << duration1.count() << " ms" << std::endl;
  }


  CUDA_RT_CALL(cudaGetLastError());
  cudaDeviceSynchronize();
  cpuThread.join();

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> duration = end - start;

  std::cout << "Totol Back: " << duration.count() << " ms" << std::endl;

  printf("BC_Back x SBR_Back  dQ:\n");
  printDeviceMatrixV2(dQ, ldQ, 3, 3);

  printDeviceMatrixV2(dQ + (m - 3) + (n - 3) * ldQ, ldQ, 3, 3);

  launchKernel_copyMatrix(gridDim, blockDim, m, n, dQ, ldQ, dW, ldW);
  checkOrthogonality(cublas_handle, m, n, dQ, ldQ, dW, ldW, dwork);


  double done = 1.0;
  double dzero = 0.0;
  double dnegone = -1.0;

  startTimer();
  start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(dwork, Z, sizeof(double) * n*n, cudaMemcpyHostToDevice);
  cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &done, dQ, ldQ,
              dwork, n, &dzero, dY, ldY);
  CUDA_RT_CALL(cudaGetLastError());
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  float finalGemm_time = stopTimer();
  duration = end - start;
  std::cout << "finalGemm_time : " << duration.count() << " ms" << std::endl;

#if CHECH_EVD_RESULT_ENABLE  
  launchKernel_copyMatrix(gridDim, blockDim, m, n, dY, ldY, dW, ldW);
  checkOrthogonality(cublas_handle, m, n, dY, ldY, dW, ldW, dwork);

  printf("finalGemm_time %ldx%ld takes %lf ms, tflops is %lf\n",
         m,
         n,
         finalGemm_time,
         2.0 * n * n * (m - 1.0 / 3.0 * n) / (finalGemm_time * 1e9));
  

  launchKernel_copyMatrix(gridDim, blockDim, m, n, dY, ldY, dQ, ldQ);       

  cudaMemcpy(mD, hS, sizeof(double) * n, cudaMemcpyHostToDevice);
  
  launchKernel_ClearMatrix(gridDimClrDu, blockDimClrDU, m, n, dA, ldA);
  dim3 blockDim4(32);
  dim3 gridDim4((m + 31) / 32);
  launch_kernel_cpyVector2ATr(gridDim4, blockDim4, n, n, dA, ldA, mD);

  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  launchKernel_copyMatrix(gridDim, blockDim, m, n, dT, m, dW, m);

  // ||OA-Q*dA*Q'||
  cublasDgemm(cublas_handle,
              CUBLAS_OP_N,
              CUBLAS_OP_T,
              m,
              n,
              m,
              &done,
              dA,
              ldA,
              dQ,
              ldQ,
              &dzero,
              dwork,
              m);

  // 2.OA=OA-Q*work
  cublasDgemm(cublas_handle,
              CUBLAS_OP_N,
              CUBLAS_OP_N,
              m,
              n,
              m,
              &dnegone,
              dQ,
              ldQ,
              dwork,
              m,
              &done,
              dW,
              m);

  double snA, snOriA;
  int incx = 1;

  cublasDnrm2(cublas_handle, m * n, dW, incx, &snA);
  cublasDnrm2(cublas_handle, m * n, dT, incx, &snOriA);

  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  cout << "Backforward err: " << snA / snOriA / m << std::endl;
#endif

  free(Z);
  free(hS);
  free(hE);

  cudaFree(mE);
  cudaFree(mD);

#endif


  cudaFree(dA);

  printf("gemm %ldx%ld takes %lf ms, tflops is %lf\n",
         m,
         n,
         g_gemm_time_ZY,
         2.0 * n * n * (m - 1.0 / 3.0 * n) / (g_gemm_time_ZY * 1e9));

  printf("syr2k %ldx%ld takes %lf ms, tflops is %lf\n",
         m,
         n,
         g_tc_ozimmu_syr2k_ZY,
         2.0 * n * n * (m - 1.0 / 3.0 * n) / (g_tc_ozimmu_syr2k_ZY * 1e9));

  printf("qr %ldx%ld takes %lf ms, tflops is %lf\n",
         m,
         n,
         g_panelQR_time_ZY,
         2.0 * n * n * (m - 1.0 / 3.0 * n) / (g_panelQR_time_ZY * 1e9));

  printf("sy2sb %ldx%ld takes %lf ms, tflops is %lf\n",
         m,
         n,
         g_sy2sb_time,
         2.0 * n * n * (m - 1.0 / 3.0 * n) / (g_sy2sb_time * 1e9));

  printf("Bugle chasing %ldx%ld takes %lf ms, tflops is %lf\n",
         m,
         n,
         g_bugle_chasing_kernel_time,
         2.0 * n * n * (m - 1.0 / 3.0 * n) / (g_bugle_chasing_kernel_time * 1e9));

#if USE_LAPACKE
  printf("DC %ldx%ld takes %lf ms, tflops is %lf\n",
         m,
         n,
         lapackestedcTime,
         2.0 * n * n * (m - 1.0 / 3.0 * n) / (lapackestedcTime * 1e9));
#endif

  printf("Bugle chasing Compute Q %ldx%ld takes %lf ms, tflops is %lf\n",
         m,
         n,
         BC_Back_time,
         2.0 * n * n * (m - 1.0 / 3.0 * n) / (BC_Back_time * 1e9));


  float ms = g_sy2sb_time + g_bugle_chasing_kernel_time;
  printf("sy2tr %ldx%ld takes %lf ms, tflops is %lf\n",
         m,
         n,
         ms,
         (4.0 * n * n * n / 3.0) / (ms * 1e9));

}
