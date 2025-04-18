

// #include <cuda_runtime.h>

// #include <cstdlib>

#include "fileOpTool.h"
#include "myBase.h"
// #include <__clang_cuda_runtime_wrapper.h>

static __inline__ __device__ double warpAllReduceSum(double val)
{
  for (int mask = warpSize / 2; mask > 0; mask /= 2)
  {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

static __inline__ __device__ void warpAllReduceSumArray(double *val, int size)
{
  for (int mask = warpSize / 2; mask > 0; mask /= 2)
  {
    for (int i = 0; i < size; i++)
    {
      val[i] += __shfl_xor_sync(0xffffffff, val[i], mask);
    }
  }
}

static __inline__ __device__ void warpAllReduceSumArrayV2(double *val, int size)
{

  for (int i = 0; i < size; i++)
  {
    val[i] = (double)__reduce_add_sync(0xffffffff, (unsigned int)val[i]);
  }

}

static __inline__ __device__ double warpAllReduceSumV2(double val, int ThreadCount = 32)
{
  for (int mask = ThreadCount / 2; mask > 0; mask /= 2)
  {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

#define U_COUNT 8
#define U_LEN_PROC_1TIME (U_COUNT * 32)

#define MAX_WARP_COUNT 24

#define SYNC_THREAD_NUM (32/U_COUNT)

#define U_COL_EXRTERN_COUNT 90

extern __shared__ double externSM[];
__global__ void BC_kernel_computerQ_1Col_V8_10_noBandU(int n,
                                            int perBlockN,
                                            int largeBlockNum,
                                            int sweepCount,
                                            int lastSweepUCount,
                                            double *dCU,
                                            long ldCU,
                                            double *dQ,
                                            long ldQ)
{
  double* sU2 = externSM;

  __shared__ double stailQ[MAX_WARP_COUNT*U_COL_EXRTERN_COUNT];

  __shared__ double stailQW[MAX_WARP_COUNT*U_COL_EXRTERN_COUNT];


  __shared__ double sTData[MAX_WARP_COUNT*32];

  double rQ[U_COUNT]; 

  double4 *rQ4 = (double4*)rQ;

  int bInx = blockIdx.x;
  if(bInx < largeBlockNum)
  {
    perBlockN += 1;
    dQ = dQ + bInx * perBlockN * ldQ;
  }else
  {
    dQ = dQ + (bInx *perBlockN + largeBlockNum) * ldQ;
  }

  int i = threadIdx.x;
  int j = threadIdx.y;


  int sweepIndex;

  int totalU; 

  long sweepBaseRow;
  long indexU = 0;

  #pragma unroll
  for (sweepIndex = 0; sweepIndex < sweepCount; sweepIndex++)
  {
    sweepBaseRow = (sweepCount - sweepIndex - 1) * U_LEN_PROC_1TIME;
    // totalU = (U_LEN_PROC_1TIME - 2) + sweepIndex * U_LEN_PROC_1TIME; 

    totalU = lastSweepUCount + sweepIndex * U_LEN_PROC_1TIME; 

    indexU = 0; 

    #pragma unroll
    for (;totalU > 0;)
    {
      __syncthreads();


      #pragma unroll
      for(int k =j; k<U_COL_EXRTERN_COUNT;k += MAX_WARP_COUNT)
      {
        #pragma unroll
        for(int t =0; t<U_COUNT;t++)
        {
          sU2[k*U_LEN_PROC_1TIME+i + t *32] = 0.0;
          if(k < totalU)
          {
            sU2[k*U_LEN_PROC_1TIME+i + t *32] = dCU[(indexU+k)*ldCU + sweepBaseRow + 1 + k + i*U_COUNT +t];
            // sU2[k*U_LEN_PROC_1TIME+i + t *32] = dCU[(indexU+k)*U_LEN_PROC_1TIME + i +t*32];
          }
        }
      }

      __syncthreads();

      for (int k = j; k < perBlockN; k += MAX_WARP_COUNT)
      {

        double4 *tmpDQ4 = (double4*)(dQ+k * ldQ + sweepBaseRow);
        #pragma unroll
        for (int t = 0; t < U_COUNT/4; t++)
        {
          rQ4[t] = tmpDQ4[i*U_COUNT/4 + t];
        }

        __syncwarp();

        #pragma unroll
        for (int t = i; t < U_COL_EXRTERN_COUNT; t +=32)
        {
          stailQ[j*U_COL_EXRTERN_COUNT + t] = dQ[k * ldQ + sweepBaseRow + U_LEN_PROC_1TIME + t];
        }
        

        __syncwarp();

        int h = 0;
        #pragma unroll
        for (; h < U_COL_EXRTERN_COUNT; h++)
        {

          if (0 != i)
          {
            sTData[j*32 + i] = rQ[0];
          }else
          {
            stailQW[j*U_COL_EXRTERN_COUNT+h] = rQ[0];
          }

          __syncwarp();

          #pragma unroll
          for (int t = 0; t < U_COUNT-1; t++)
          {
            rQ[t] = rQ[t+1];
          }

          if(31 != i)
          {
            rQ[U_COUNT-1] = sTData[j*32 + i+1];
          }else{

            rQ[U_COUNT-1] = stailQ[j*U_COL_EXRTERN_COUNT+h];
          }
          __syncwarp();

          double nux = 0.0;

          #pragma unroll
          for (int t = 0; t < U_COUNT; t++)
          {
            nux += sU2[h*U_LEN_PROC_1TIME+i + t * 32] * rQ[t];
          }

          nux = warpAllReduceSumV2(nux, SYNC_THREAD_NUM);

          #pragma unroll
          for (int t = 0; t < U_COUNT; t++)
          {
            rQ[t] -= nux * sU2[h*U_LEN_PROC_1TIME+i+t * 32];
          }

        }

        #pragma unroll
        for (int t = i; t < U_COL_EXRTERN_COUNT; t +=32)
        {
          // stailQ[j*U_COL_EXRTERN_COUNT + t] = dQ[k * ldQ + sweepBaseRow + U_LEN_PROC_1TIME + t];
          dQ[k * ldQ + sweepBaseRow + t] = stailQW[j*U_COL_EXRTERN_COUNT + t];
        }

        
        tmpDQ4 = (double4*)(dQ+k * ldQ + sweepBaseRow + h);
        #pragma unroll
        for (int t = 0; t < U_COUNT/4; t++)
        {
          tmpDQ4[i*U_COUNT/4 + t] = rQ4[t];
        }
      }

      indexU += U_COL_EXRTERN_COUNT;
      totalU -= U_COL_EXRTERN_COUNT;

      sweepBaseRow += U_COL_EXRTERN_COUNT;

      __syncthreads();
    }


  }
}

int my_BC_back_trans_v8_10_noBandU(double *Q, long ldQ, double *U, long ldU, long n, int b, cudaStream_t stream = NULL)
{

  int sweepCount = (n - 1 - 1 + (U_LEN_PROC_1TIME - 1)) / (U_LEN_PROC_1TIME);

  int lastSweepUCount = n - ((sweepCount-1)*U_LEN_PROC_1TIME+1)-1;

  long countU = 0;
  for (int i = 0, tmp = lastSweepUCount; i <sweepCount; i++, tmp+=U_LEN_PROC_1TIME)
  {
    int tmp2 = (tmp+U_COL_EXRTERN_COUNT-1)/U_COL_EXRTERN_COUNT*U_COL_EXRTERN_COUNT;
    countU += tmp2;
  }


  ssize_t shareDyMem = U_COL_EXRTERN_COUNT* U_LEN_PROC_1TIME *8; 
  cudaFuncSetAttribute(BC_kernel_computerQ_1Col_V8_10_noBandU,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       shareDyMem);


  dim3 dimBlock(32, MAX_WARP_COUNT, 1);
#if 1
  int dev                = 0;


  int numBlocksPerSm = 0;
  int numThreads = 32 * MAX_WARP_COUNT;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm,
                                                                  BC_kernel_computerQ_1Col_V8_10_noBandU,
                                                                  numThreads,
                                                                  shareDyMem);
  if (err != cudaSuccess)
  {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    return -1;
  }
 
  int blockNum = numBlocksPerSm * deviceProp.multiProcessorCount;

  int perBlockN = n / blockNum;
  int largeBlockNum = n % blockNum;


  startTimer();
  void *kernelArgs[] = {(void *)&n,
                        (void *)&perBlockN,
                        (void *)&largeBlockNum,
                        (void *)&sweepCount,
                        (void *)&lastSweepUCount,
                        (void *)&U,
                        (void *)&ldU,
                        (void *)&Q,
                        (void *)&ldQ};

  dim3 dimGrid(blockNum, 1, 1);
  cudaLaunchCooperativeKernel((void *)BC_kernel_computerQ_1Col_V8_10_noBandU,
                              dimGrid,
                              dimBlock,
                              kernelArgs,
                              shareDyMem,
                              stream);
#else

  BC_kernel_computerQ_1Col_V8<<<114, dimBlock>>>(n, b, dCU, countU, dEQ, ldEQ, sweepCount, 114);
#endif

  float g_SVD_BC_BackTans_Time = stopTimer();


  printf("global BC Back %ldx%ld takes %lf ms, tflops is %lf\n",
         n,
         n,
         g_SVD_BC_BackTans_Time,
         (2.0 * n * n * n ) / (g_SVD_BC_BackTans_Time * 1e9));

  return 0;
}
