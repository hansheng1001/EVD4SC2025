#include <cuda_fp16.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <curand.h>

template <typename T>
void generateUniformMatrix(T *dA, long int m, long int n);

template <>
void generateUniformMatrix(double *dA, long int m, long int n)
{
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  int seed = 3000;
  curandSetPseudoRandomGeneratorSeed(gen, seed);

  curandGenerateUniformDouble(gen, dA, long(m * n));
}

template <>
void generateUniformMatrix(float *dA, long int m, long int n)
{
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  int seed = 3000;
  curandSetPseudoRandomGeneratorSeed(gen, seed);
  curandGenerateUniform(gen, dA, long(m * n));
}

__global__ static void
matrixCpyF2H(long int m, long int n, float *a, long int lda, half *b, long int ldb)
{
  long int i = threadIdx.x + blockDim.x * blockIdx.x;
  long int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i < m && j < n)
  {
    b[i + j * ldb] = __half2float(a[i + j * lda]);
  }
}

template <>
void generateUniformMatrix(half *dA, long int m, long int n)
{
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  int seed = 3000;
  curandSetPseudoRandomGeneratorSeed(gen, seed);

  float *_dA;
  cudaMalloc((void **)&_dA, sizeof(float) * m * n);

  curandGenerateUniform(gen, _dA, long(m * n));

  dim3 blockDim(32, 32);
  dim3 gridDim((m + 32 - 1) / 32, (n + 32 - 1) / 32);
  matrixCpyF2H<<<gridDim, blockDim>>>(m, n, _dA, m, dA, m);

  cudaFree(_dA);
}

template <typename T>
void generateNormalMatrix(T *dA, long int m, long int n, T mean, T stddev);

template <>
void generateNormalMatrix(double *dA, long int m, long int n, double mean, double stddev)
{
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  int seed = 3000;
  curandSetPseudoRandomGeneratorSeed(gen, seed);

  curandGenerateNormalDouble(gen, dA, long(m * n), mean, stddev);
}

template <>
void generateNormalMatrix(float *dA, long int m, long int n, float mean, float stddev)
{
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  int seed = 3000;
  curandSetPseudoRandomGeneratorSeed(gen, seed);

  curandGenerateNormal(gen, dA, long(m * n), mean, stddev);
}

template <typename T>
void printDeviceMatrixV2(T *dA, long ldA, long rows, long cols)
{
  T matrix;

  for (long i = 0; i < rows; i++)
  {
    for (long j = 0; j < cols; j++)
    {
      cudaMemcpy(&matrix, dA + i + j * ldA, sizeof(T), cudaMemcpyDeviceToHost);

      printf("%.14f ", matrix); 
    }
    printf("\n");
  }
}

template void printDeviceMatrixV2(double *dA, long ldA, long rows, long cols);
template void printDeviceMatrixV2(float *dA, long ldA, long rows, long cols);

template <>
void printDeviceMatrixV2(half *dA, long ldA, long rows, long cols)
{
  half matrix;

  for (long i = 0; i < rows; i++)
  {
    for (long j = 0; j < cols; j++)
    {
      cudaMemcpy(&matrix, dA + i + j * ldA, sizeof(half), cudaMemcpyDeviceToHost);
      float f = __half2float(matrix);
      printf("%.14f ", f);
    }
    printf("\n");
  }
}

std::vector<std::vector<double>> readMatrixFromFile(const std::string &fileName)
{
  std::vector<std::vector<double>> matrix;

  std::ifstream file(fileName);

  if (file.is_open())
  {
    std::string line;
    while (getline(file, line))
    {
      std::vector<double> row;
      std::stringstream ss(line);
      std::string cell;

      while (getline(ss, cell, ','))
      {
        row.push_back(std::stod(cell));
      }

      matrix.push_back(row);
    }

    file.close();
    std::cout << "Matrix read from " << fileName << std::endl;
  }
  else
  {
    std::cout << "Failed to open file: " << fileName << std::endl;
  }

  return matrix;
}

void fillMatrix(double *matrix, std::vector<std::vector<double>> &data)
{
  long rows = data.size();
  long cols = data[0].size();


  for (long i = 0; i < cols; i++)
  {
    for (long j = 0; j < rows; j++)
    {
      matrix[i * rows + j] = data[j][i];
    }
  }
}

void printMatrix(double *matrix, long ldA, long rows, long cols)
{
  for (long i = 0; i < rows; i++)
  {
    for (long j = 0; j < cols; j++)
    {

      printf("%0.14f ", matrix[j * ldA + i]); 
    }
    printf("\n");
  }
}

void printDeviceMatrixV2Int(int *dA, long ldA, long rows, long cols)
{
  int matrix;

  for (long i = 0; i < rows; i++)
  {
    for (long j = 0; j < cols; j++)
    {
      cudaMemcpy(&matrix, dA + i + j * ldA, sizeof(int), cudaMemcpyDeviceToHost);

      printf("%d ", matrix); 
    }
    printf("\n");
  }
}

