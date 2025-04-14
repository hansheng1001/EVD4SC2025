#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <curand.h>

void generateUniformMatrix(double *dA, long int m, long int n)
{
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  int seed = 3000;
  curandSetPseudoRandomGeneratorSeed(gen, seed);
  curandGenerateUniformDouble(gen, dA, long(m * n));
}

void generateUniformMatrixFloat(float *dA, long int m, long int n)
{
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  int seed = 3000;
  curandSetPseudoRandomGeneratorSeed(gen, seed);
  curandGenerateUniform(gen, dA, long(m * n));
}

void generateNormalMatrix(double *dA, long int m, long int n, double mean, double stddev)
{
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  int seed = 3000;
  curandSetPseudoRandomGeneratorSeed(gen, seed);

  curandGenerateNormalDouble(gen, dA, long(m * n), mean, stddev);
}

void generateNormalMatrixfloat(float *dA, long int m, long int n, float mean, float stddev)
{
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  int seed = 3000;
  curandSetPseudoRandomGeneratorSeed(gen, seed);

  curandGenerateNormal(gen, dA, long(m * n), mean, stddev);
}

std::vector<std::vector<double>> readMatrixFromFile(
    const std::string &fileName)
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

// template<typename T>
// void printDeviceMatrix(T *dA, long rows, long cols) {

//   T *matrix;
//   matrix = (T *)malloc(sizeof(T) * m * n);

//   cudaMemcpy(matrix, dA, sizeof(T) * m * n, cudaMemcpyDeviceToHost);

//   for (long i = 0; i < rows; i++) {
//     for (long j = 0; j < cols; j++) {
//       // printf("%f ", matrix[i * cols + j]);//按行存储优先
//       printf("%10.4f", matrix[j * rows + i]);  // 按列存储优先
//     }
//     printf("\n");
//   }

//   free(matrix);
// }