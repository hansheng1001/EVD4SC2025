
#pragma once

// #include "fileOpTool.h"

static __inline__ __device__ double warpAllReduceSum(double val)
{
    for (int mask = warpSize / 2; mask > 0; mask /= 2)
    {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// __device__ volatile int com[8192] = {0};
// __device__ volatile int com[32768] = {0};
__device__ volatile int com[65536] = {0};

template <int BandWidth>
// __device__ void chasing_kernel_one_timeV7(int n, int b, double *subA, int ldSubA, int startRow)
__global__ void chasing_kernel_one_timeV7(int n, int b, double *subA, int ldSubA)
{

    int Nx = blockDim.x;
    int Ny = blockDim.y;


    if (BandWidth != b)
    {
        return;
    }

    int warpGroupThdCount = Ny / 2;

    int bInx = blockIdx.x;

    int i = threadIdx.x;
    int j = threadIdx.y;

    // #if MY_DEBUG
    if ((0 == i) && (0 == j))
    {
        // printf("block[%d] [%d][%d] come line=%d.\n", bInx, i, j, __LINE__);
        printf("\b");
    }
    // #endif



    __shared__ double u[BandWidth];

    __shared__ double S1[BandWidth * BandWidth]; 
    __shared__ double S2[BandWidth * BandWidth]; 
    int ldSS = b;

    double nu;
    double utx;

    int opColB1;
    int opRowB1;


    // int opRow = bInx + 1;
    long opRow = bInx + 1;

    int rowB1 = min(b, (int)(n - opRow)); 
    int colB1 = 1;


    double *B1 = subA + colB1 + (opRow - colB1) * ldSubA;


    while ((0 != bInx) && (opRow + 2 * b > com[bInx - 1]))
    {
        // if ((1 == bInx) && (0 == i) && (0 == j))
        // {
        //     printf("Begin block[%2d] [%d][%d] come line=%d, com[%d] = %d.\n",
        //            bInx, i, j, __LINE__, bInx - 1, com[bInx - 1]);
        // }
        continue;
    }


#pragma unroll
    for (opColB1 = j; opColB1 < colB1; opColB1 += Ny)
    {
#pragma unroll
        for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
        {

            S1[opRowB1 + opColB1 * ldSS] = B1[(opRowB1 - opColB1) + opColB1 * ldSubA];
        }
    }


    int rowS = 0;
    int colS = 0;

    double *S; 


    while (1)
    {

        if (0 != j)
        {

#pragma unroll
            for (int opColS = j - 1; opColS < colS; opColS += (Ny - 1))
            {
#pragma unroll
                for (int opRowS = i; (opColS <= opRowS) && (opRowS < rowS); opRowS += Nx)
                {

                    S[(opRowS - opColS) + opColS * ldSubA] = S2[opRowS + opColS * ldSS];
                }
            }


            colS = rowS = rowB1;
            S = subA + opRow * ldSubA;

#pragma unroll
            for (int opColS = j - 1; opColS < colS; opColS += (Ny - 1))
            {
#pragma unroll
                for (int opRowS = i; (opColS <= opRowS) && (opRowS < rowS); opRowS += Nx)
                {

                    S2[opRowS + opColS * ldSS] = S[(opRowS - opColS) + opColS * ldSubA];


                    S2[opColS + opRowS * ldSS] = S2[opRowS + opColS * ldSS];
                }
            }

            // if ((0 == i) && (1 == j))
            // {
            //     printf("block[%d] [%d][%d] come line=%d.\n", bInx, i, j, __LINE__);
            //     printf("\b");
            // }
        }
        else
        {

#pragma unroll
            for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
            {
                //  u[opRowB1] = B1[opRowB1][0]
                u[opRowB1] = S1[opRowB1];
            }

            __syncwarp();


            nu = 0.0;

#pragma unroll
            for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
            {
                //  u[opRowB1] = B1[opRowB1][0]
                nu += u[opRowB1] * u[opRowB1];
            }


            double norm_x_squre = warpAllReduceSum(nu);
            double norm_x = sqrt(norm_x_squre);


            double scale = 1.0 / norm_x;
#pragma unroll
            for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
            {
                //  u[opRowB1] = B1[opRowB1][0]
                u[opRowB1] *= scale;
            }

            __syncwarp();


            if (0 == i)
            {
                double u1 = u[0];

                u[0] += (u1 >= 0) ? 1 : -1;


            }

            __syncwarp();


            scale = 1 / (sqrt(abs(u[0])));
#pragma unroll
            for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
            {
                //  u[opRowB1] = B1[opRowB1][0]
                u[opRowB1] *= scale;
            }


            colS = rowS = rowB1;
            S = subA + opRow * ldSubA;
        }

        __syncthreads();


#pragma unroll
        for (opColB1 = j; opColB1 < colB1; opColB1 += Ny)
        {
            nu = 0.0;

#pragma unroll
            for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
            {
                nu += u[opRowB1] * S1[opRowB1 + opColB1 * ldSS];
            }

            utx = warpAllReduceSum(nu);


#pragma unroll
            for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
            {
                S1[opRowB1 + opColB1 * ldSS] -= utx * u[opRowB1];
            }

            __syncwarp();
        }

        __syncthreads();


        if (j < warpGroupThdCount)
        {

#pragma unroll
            for (opColB1 = j; opColB1 < colB1; opColB1 += warpGroupThdCount)
            {
#pragma unroll
                for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
                {
                    B1[(opRowB1 - opColB1) + opColB1 * ldSubA] = S1[opRowB1 + opColB1 * ldSS];


                }
            }

            opRow += rowB1;                   
            rowB1 = min(b, (int)(n - opRow)); 
            colB1 = colS;
            B1 = subA + colB1 + (opRow - colB1) * ldSubA;

  
            while ((0 != bInx) && (opRow + 2 * b > com[bInx - 1]))
            {
                // if ((1 == bInx) && (0 == i) && (0 == j))
                // {
                //     printf("Begin block[%2d] [%d][%d] come line=%d, com[%d] = %d.\n",
                //            bInx, i, j, __LINE__, bInx - 1, com[bInx - 1]);
                // }
                continue;
            }


#pragma unroll
            for (opColB1 = j; opColB1 < colB1; opColB1 += warpGroupThdCount)
            {
#pragma unroll
                for (opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
                {
                    // SS[opRowB1][opColB1] = B1[opRowB1][opColB1] = B2[opColB1][opRowB1]
                    S1[opRowB1 + opColB1 * ldSS] = B1[(opRowB1 - opColB1) + opColB1 * ldSubA];
                }
            }

            // if ((0 == i) && (0 == j))
            // {
            //     printf("block[%d] [%d][%d] come line=%d.\n", bInx, i, j, __LINE__);
            //     printf("\b");
            // }
        }
        else
        {
            
#pragma unroll
            for (int opColS = j - warpGroupThdCount; opColS < colS; opColS += warpGroupThdCount)
            {
                nu = 0.0;
   
#pragma unroll
                for (int opRowS = i; opRowS < rowS; opRowS += Nx)
                {
                    nu += u[opRowS] * S2[opRowS + opColS * ldSS];
                }

                utx = warpAllReduceSum(nu);

     
#pragma unroll
                for (int opRowS = i; opRowS < rowS; opRowS += Nx)
                {
                    S2[opRowS + opColS * ldSS] -= utx * u[opRowS];
                }

                __syncwarp();
            }

           
            opRow += rowB1;                   
            rowB1 = min(b, (int)(n - opRow)); 
            colB1 = colS;
            B1 = subA + colB1 + (opRow - colB1) * ldSubA;


        }

        __syncthreads();
#pragma unroll
        for (int opRowS = j; opRowS < rowS; opRowS += Ny)
        {
            nu = 0.0;

#pragma unroll
            for (int opColS = i; opColS < colS; opColS += Nx)
            {
                nu += u[opColS] * S2[opRowS + opColS * ldSS];
            }

            utx = warpAllReduceSum(nu);


#pragma unroll
            for (int opColS = i; opColS < colS; opColS += Nx)
            {
                S2[opRowS + opColS * ldSS] -= utx * u[opColS];
            }

            __syncwarp();
        }

        __syncthreads();


        if ((0 == i) && (0 == j))
        {
            com[bInx] = opRow;
        }

#if MY_DEBUG
        if ((0 == i) && (0 == j))
        {
            printf("block[%d] [%d][%d] come line=%d, opRow=%d, rowB1 =%d, colB1=%d.\n",
                   bInx, i, j, __LINE__, opRow, rowB1, colB1);
            printf("\b");
        }
#endif


#pragma unroll
        for (opRowB1 = j; opRowB1 < rowB1; opRowB1 += Ny)
        {
            nu = 0.0;

#pragma unroll
            for (opColB1 = i; opColB1 < colB1; opColB1 += Nx)
            {
                nu += u[opColB1] * S1[opRowB1 + opColB1 * ldSS];
            }



            utx = warpAllReduceSum(nu);


#pragma unroll
            for (opColB1 = i; opColB1 < colB1; opColB1 += Nx)
            {
                S1[opRowB1 + opColB1 * ldSS] -= utx * u[opColB1];
            }


            __syncwarp();
        }

        __syncthreads();


        if (rowB1 <= 1)
        {
            break;
        }

    }


#pragma unroll
    for (int opColS = j; opColS < colS; opColS += Ny)
    {
#pragma unroll
        for (int opRowS = i; (opColS <= opRowS) && (opRowS < rowS); opRowS += Nx)
        {

            S[(opRowS - opColS) + opColS * ldSubA] = S2[opRowS + opColS * ldSS];
        }
    }


#pragma unroll
    for (int opColB1 = j; opColB1 < colB1; opColB1 += Ny)
    {
#pragma unroll
        for (int opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx)
        {

            B1[(opRowB1 - opColB1) + opColB1 * ldSubA] = S1[opRowB1 + opColB1 * ldSS];


        }
    }

    __syncthreads();


    if ((0 == i) && (0 == j))
    {
        com[bInx] = n + 4 * b;
    }

#if MY_DEBUG
    if ((0 == i) && (1 == j))
    {
        printf("block[%d] [%d][%d] come line=%d.\n", bInx, i, j, __LINE__);
        printf("\b");
    }
#endif
}

