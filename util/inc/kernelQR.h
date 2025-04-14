
#pragma once

static __inline__ __device__ double warpAllReduceSum(double val)
{
    for (int mask = warpSize / 2; mask > 0; mask /= 2)
    {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <long M, long N>
__global__ void my_hou_kernel(long m, long n, double *A, long lda, double *R, long ldr)
{
    // printf("come 19\n");
    // // cudaDeviceSynchronize();
    // __syncthreads();


    long mm = min(m - blockIdx.x * M, M);


    if (0 >= mm)
    {
        return;
    }

    A = A + blockIdx.x * M;
    R = R + blockIdx.x * N;

    // printf("come 28\n");
    // // cudaDeviceSynchronize();
    // __syncthreads();


    long nn = min(N, n);


    // long i = blockIdx.x * blockDim.x + threadIdx.x;
    // long j = blockIdx.y * blockDim.y + threadIdx.y;
    long i = threadIdx.x;
    long j = threadIdx.y;


    // __shared__ double AA[mm * nn], RR[nn];
    __shared__ double AA[M * N], RR[N];
    long ldaa = mm;


    long rowDataNum = (mm + (blockDim.x - 1)) / blockDim.x;
    long colDataNum = (nn + (blockDim.y - 1)) / blockDim.y;

    // double acc[rowDataNum];
    double acc[8];

    for (long k = 0; k < rowDataNum; k++)
    {
        if (i + k * blockDim.x < mm)
        {
            AA[i + k * blockDim.x + j * ldaa] = A[i + k * blockDim.x + j * lda];
            AA[i + k * blockDim.x + (j + 16) * ldaa] = A[i + k * blockDim.x + (j + 16) * lda];
        }
    }


    __syncthreads();


    for (long cols = 0; cols < nn; cols++)
    {

        double nu = 0.0;
        if (j == cols % blockDim.y)
        {

#pragma unroll
            for (long k = 0; k < rowDataNum; k++)
            {
                acc[k] = 0.0;

                if (i + k * blockDim.x < mm && i + k * blockDim.x >= cols)
                {
                    acc[k] = AA[i + k * blockDim.x + cols * ldaa] * AA[i + k * blockDim.x + cols * ldaa];
                }
                nu += acc[k];
            }


            double norm_x_squre = warpAllReduceSum(nu);
            double norm_x = sqrt(norm_x_squre);


            double scale = 1.0 / norm_x;
#pragma unroll
            for (long k = 0; k < rowDataNum; k++)
            {
                if (i + k * blockDim.x < mm && i + k * blockDim.x >= cols)
                {
                    AA[i + k * blockDim.x + cols * ldaa] *= scale;
                }
            }

            __syncwarp();


            if (0 == i)
            {
                double u1 = AA[cols + cols * mm];

                AA[cols + cols * ldaa] += (u1 >= 0) ? 1 : -1;


                RR[cols] = (u1 >= 0) ? -norm_x : norm_x;
            }

            __syncwarp();


            scale = 1 / (sqrt(abs(AA[cols + cols * ldaa])));
#pragma unroll
            for (long k = 0; k < rowDataNum; k++)
            {
                if (i + k * blockDim.x < mm && i + k * blockDim.x >= cols)
                {
                    AA[i + k * blockDim.x + cols * ldaa] *= scale;
                }
            }
        }

        __syncthreads();

        for (int h = 0; h < colDataNum; h++)
        {
            double nu = 0.0;
            long opCols = j + h * blockDim.y;


            if (cols < opCols && opCols <= nn)
            {

#pragma unroll
                for (long k = 0; k < rowDataNum; k++)
                {
                    acc[k] = 0.0;
        
                    if (i + k * blockDim.x < mm && i + k * blockDim.x >= cols)
                    {
                        acc[k] = AA[i + k * blockDim.x + cols * ldaa] * AA[i + k * blockDim.x + opCols * ldaa];
                    }
                    nu += acc[k];
                }
                double utx = warpAllReduceSum(nu);

  
#pragma unroll
                for (long k = 0; k < rowDataNum; k++)
                {

                    if (i + k * blockDim.x < mm && i + k * blockDim.x >= cols)
                    {
                        AA[i + k * blockDim.x + opCols * ldaa] -= utx * AA[i + k * blockDim.x + cols * ldaa];
                    }
                }
                __syncwarp();
            }
        }
    }

    __syncthreads();

    long rRowDataNum = (nn + (blockDim.x - 1)) / blockDim.x;
    for (int h = 0; h < colDataNum; h++)
    {
        long opCols = j + h * blockDim.y;

        if (opCols >= nn)
            continue;

#pragma unroll
        for (long k = 0; k < rRowDataNum; k++)
        {
            if (i + k * blockDim.x < opCols)
            {
                R[i + k * blockDim.x + opCols * ldr] = AA[i + k * blockDim.x + opCols * ldaa];
                AA[i + k * blockDim.x + opCols * ldaa] = 0.0;
            }
            else if (i + k * blockDim.x > opCols)
            {
                R[i + k * blockDim.x + opCols * ldr] = 0.0;
            }
            else
            {

                R[opCols + opCols * ldr] = RR[opCols];
            }
        }
    }

    

    double q[8 * 2];
    for (int h = 0; h < colDataNum; h++)
    {

        long opCols = j + h * blockDim.y;

        if (opCols >= nn)
            continue;

        for (long k = 0; k < rowDataNum; k++)
        {
            if (i + k * blockDim.x == opCols)
            {
                q[k] = 1.0;
            }
            else
            {
                q[k] = 0.0;
            }
        }

        __syncwarp();

        for (int cols = nn - 1; cols >= 0; cols--)
        {


            if (opCols >= cols)
            {

                double nu = 0.0;
                for (long k = 0; k < rowDataNum; k++)
                {
                    acc[k] = 0.0;
                    if (i + k * blockDim.x < mm)
                    {
                        acc[k] = AA[i + k * blockDim.x + cols * ldaa] * q[k];
                    }
                    nu += acc[k];
                }

                double utq = warpAllReduceSum(nu);

 
                for (long k = 0; k < rowDataNum; k++)
                {
                    if (i + k * blockDim.x < mm)
                    {
                        q[k] -= utq * AA[i + k * blockDim.x + cols * ldaa];
                    }
                }

                __syncwarp();
            }
        }

        for (long k = 0; k < rowDataNum; k++)
        {
            if (i + k * blockDim.x < mm)
            {
                A[i + k * blockDim.x + opCols * lda] = q[k];
            }
        }
    }
}
