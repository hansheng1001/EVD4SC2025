#include <stdio.h>
#include "kernelOther.h"

__global__ void clearMatrix(long m, long n, double *A, long ldA)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("come %d\n", __LINE__);
    // __syncthreads();

    if (i < m && j < n)
    {
        A[i + j * ldA] = 0.0;
    }
}

void launchKernel_ClearMatrix(dim3 gridDim, dim3 blockDim, long m, long n, double *A, long ldA)
{
    clearMatrix<<<gridDim, blockDim>>>(m, n, A, ldA);
}

static __global__ void kernel_setMetrixTrValue(long m, long n, double *A, long ldA, double v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("come %d\n", __LINE__);
    // __syncthreads();
    if ((i < m) && (i < n))
    {
        A[i + i * ldA] = v;
    }
}

void launchKernel_setMetrixTrValue(dim3 gridDim, dim3 blockDim, long m, long n, double *A, long ldA, double v)
{
    kernel_setMetrixTrValue<<<gridDim, blockDim>>>(m, n, A, ldA, v);
}

__global__ void copyMatrixL2U(long n, double *A, long ldA)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("come %d\n", __LINE__);
    // __syncthreads();

    if (i < n && j < n)
    {
        if (j > i)
            A[i + j * ldA] = A[j + i * ldA];
    }
}

void launchKernel_CpyMatrixL2U(dim3 gridDim, dim3 blockDim, long n, double *A, long ldA)
{
    copyMatrixL2U<<<gridDim, blockDim>>>(n, A, ldA);
}

__global__ void copyAndClear(long m, long n, double *srcM, long lds,
                             double *dstM, long ldd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n)
    {
        dstM[i + j * ldd] = srcM[i + j * lds];
        srcM[i + j * lds] = 0.0;
    }
}

void launchKernel_copyAndClear(dim3 gridDim, dim3 blockDim, long m, long n, double *srcM, long lds,
                               double *dstM, long ldd)
{
    copyAndClear<<<gridDim, blockDim>>>(m, n, srcM, lds, dstM, ldd);
}

__global__ void IminusQ(long m, long n, double *Q, long ldq)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("come %d, %d, %d,\n", __LINE__, i, j);
    // __syncthreads();

    if (i < m && j < n)
    {
        if (i == j)
        {
            Q[i + j * ldq] = 1.0 - Q[i + j * ldq];
        }
        else
        {
            Q[i + j * ldq] = -Q[i + j * ldq];
        }

        // printf("come %d, %d, %d,\n", __LINE__, i, j);
        // __syncthreads();
    }
}

void launchKernel_IminusQ(dim3 gridDim, dim3 blockDim, long m, long n, double *Q, long ldq)
{
    IminusQ<<<gridDim, blockDim>>>(m, n, Q, ldq);
}

__global__ void AminusB(long m, long n, double *A, long ldA, double *B, long ldB)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("come %d, %d, %d,\n", __LINE__, i, j);
    // __syncthreads();

    if (i < m && j < n)
    {

        A[i + j * ldA] -= B[i + j * ldB];

        // printf("come %d, %d, %d,\n", __LINE__, i, j);
        // __syncthreads();
    }
}

void launchKernel_AminusB(dim3 gridDim, dim3 blockDim, long m, long n, double *A, long ldA, double *B, long ldB)
{
    AminusB<<<gridDim, blockDim>>>(m, n, A, ldA, B, ldB);
}

__global__ void AbsAminusAbsB(long m, long n, double *A, long ldA, double *B, long ldB)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("come %d, %d, %d,\n", __LINE__, i, j);
    // __syncthreads();

    if (i < m && j < n)
    {
        // A[i + j * ldA] = abs(A[i + j * ldA]);
        double t = abs(A[i + j * ldA]);

        A[i + j * ldA] = t - abs(B[i + j * ldB]);

        // printf("come %d, %d, %d,\n", __LINE__, i, j);
        // __syncthreads();
    }
}

void launchKernel_AbsAminusAbsB(dim3 gridDim, dim3 blockDim, long m, long n, double *A, long ldA, double *B, long ldB)
{
    AbsAminusAbsB<<<gridDim, blockDim>>>(m, n, A, ldA, B, ldB);
}

__global__ void copyMatrix(long m, long n, double *srcM, long lds,
                           double *dstM, long ldd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n)
    {
        dstM[i + j * ldd] = srcM[i + j * lds];
    }
}

void launchKernel_copyMatrix(dim3 gridDim, dim3 blockDim, long m, long n, double *srcM, long lds,
                             double *dstM, long ldd)
{
    copyMatrix<<<gridDim, blockDim>>>(m, n, srcM, lds, dstM, ldd);
}


__global__ void copyMatrixAToTranpB(long m, long n, double *srcM, long lds,
                                    double *dstM, long ldd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // dst[j][i] = src[i][j]
    if (i < m && j < n)
    {
        dstM[j + i * ldd] = srcM[i + j * lds];
    }
}

void launchKernel_copyMatrixAToTranpB(dim3 gridDim, dim3 blockDim, long m, long n, double *srcM, long lds,
                                      double *dstM, long ldd)
{
    copyMatrixAToTranpB<<<gridDim, blockDim>>>(m, n, srcM, lds, dstM, ldd);
}


__global__ void getU(int m, int n, double *A, int ldA, double *U, int ldU)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < m && j < n)
    {
        if (i > j)
            U[i + j * ldU] = 0;
        else
            U[i + j * ldU] = A[i + j * ldA];
    }
}

void launchKernel_getU(dim3 gridDim, dim3 blockDim, int m, int n, double *A, int ldA, double *U, int ldU)
{
    getU<<<gridDim, blockDim>>>(m, n, A, ldA, U, ldU);
}

__global__ void getLower(long m, long n, double *dA, long lda)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("come %d, %d, %d,\n", __LINE__, i, j);
    // __syncthreads();

    if (i < m && j < n)
    {
        if (i < j)
        {
            dA[i + j * lda] = 0.0;
        }
        else if (i == j)
        {
            dA[i + j * lda] = 1.0;
        }

        // printf("come %d, %d, %d,\n", __LINE__, i, j);
        // __syncthreads();
    }
}

void launchKernel_getLower(dim3 gridDim, dim3 blockDim, long m, long n, double *A, long ldA)
{
    getLower<<<gridDim, blockDim>>>(m, n, A, ldA);
}

__global__ void kernel_cpyATr2Vector(long m, long n, double *A, long ldA, double *B)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("come %d\n", __LINE__);
    // __syncthreads();
    if ((i < m) && (i < n))
    {
        B[i] = A[i + i * ldA];
    }
}

void launch_kernel_cpyATr2Vector(dim3 gridDim, dim3 blockDim,
                                 long m, long n, double *A, long ldA, double *B)
{
    kernel_cpyATr2Vector<<<gridDim, blockDim>>>(m, n, A, ldA, B);
}

__global__ void scaleMatrixA(long m, long n, double *A, long ldA, double scaler)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("come %d, %d, %d,\n", __LINE__, i, j);
    // __syncthreads();

    if (i < m && j < n)
    {

        A[i + j * ldA] *= scaler;

        // printf("come %d, %d, %d,\n", __LINE__, i, j);
        // __syncthreads();
    }
}

void launchKernel_scaleMatrixA(dim3 gridDim, dim3 blockDim, long m, long n, double *A, long ldA, double scaler)
{
    scaleMatrixA<<<gridDim, blockDim>>>(m, n, A, ldA, scaler);
}

__global__ void findAbsMaxKernel(double *d_array, double *d_max, int n)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    if (idx < n)
    {
        sdata[tid] = abs(d_array[idx]);
    }
    else
    {
        // sdata[tid] = -INFINITY; // Ensure out of bounds values do not affect max
        sdata[tid] = 0;
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write the maximum value for this block to the output array
    if (tid == 0)
    {
        d_max[blockIdx.x] = sdata[0];
    }
}

double findVectorAbsMax(double *d_array, int n)
{
    double *d_max;
    double *h_max = new double[(n + 255) / 256];
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc((void **)&d_max, blocksPerGrid * sizeof(double));

    findAbsMaxKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_array, d_max, n);

    cudaMemcpy(h_max, d_max, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);

    // Perform final reduction on the CPU
    // double max_val = -INFINITY;
    double max_val = 0;
    for (int i = 0; i < blocksPerGrid; i++)
    {
        max_val = std::max(max_val, h_max[i]);
    }

    cudaFree(d_max);
    delete[] h_max;

    return max_val;
}