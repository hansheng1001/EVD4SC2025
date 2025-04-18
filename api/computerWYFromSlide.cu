
#include "computerWYFromSlide.h"


void computerWYFromSlide(cublasHandle_t cublas_handle, long M, long N, long slideWidth, double *dW, long ldW,
                         double *dY, long ldY, double *work)
{
    long b = slideWidth;
    double done = 1.0;
    double dzero = 0.0;
    double dnegone = -1.0;

    long ldWork = M;

    for (long i = 2 * b; i <= N; i += b)
    {

        cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, i - b, b, M,
                    &done, dY, ldY, dW + (i - b) * ldW, ldW, &dzero, work, ldWork);


        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, b, i - b,
                    &dnegone, dW, ldW, work, ldWork, &done, dW + (i - b) * ldW, ldW);
    }
}