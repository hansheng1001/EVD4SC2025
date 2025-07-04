#pragma once

#include <cuda_fp16.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


template <typename T>
void generateUniformMatrix(T *dA, long int m, long int n);

template <typename T>
void generateNormalMatrix(T *dA, long int m, long int n, T mean, T stddev);

void printDeviceMatrixV2Int(int *dA, long ldA, long rows, long cols);

std::vector<std::vector<double>> readMatrixFromFile(const std::string &fileName);

void fillMatrix(double *matrix, std::vector<std::vector<double>> &data);

void printMatrix(double *matrix, long ldA, long rows, long cols);

template <typename T>
void printDeviceMatrix(T *dA, long rows, long cols)
{
  T *matrix;
  matrix = (T *)malloc(sizeof(T) * rows * cols);

  cudaMemcpy(matrix, dA, sizeof(T) * rows * cols, cudaMemcpyDeviceToHost);

  for (long i = 0; i < rows; i++)
  {
    for (long j = 0; j < cols; j++)
    {

      printf("%10.4f", matrix[j * rows + i]); 
    }
    printf("\n");
  }

  free(matrix);
}

template <typename T>
void printDeviceMatrixV2(T *dA, long ldA, long rows, long cols);

template <typename T>
void writeMatrixToCsvV2(T *dA, long ldA, long rows, long cols, const std::string &fileName)
{
  T matrix;

  std::ofstream file(fileName);

  if (file.is_open())
  {
    for (long i = 0; i < rows; i++)
    {
      for (long j = 0; j < cols; j++)
      {
        cudaMemcpy(&matrix, dA + i + j * ldA, sizeof(T), cudaMemcpyDeviceToHost);
        file << matrix;
        if ((cols - 1) != j)
        {
          file << ",";
        }
      }
      file << std::endl;
    }
    file.close();
    std::cout << "Matrix written to " << fileName << std::endl;
  }
  else
  {
    std::cout << "Failed to open file: " << fileName << std::endl;
  }
}

template <typename T>
void printAndWriteMatrixToCsvV2(T *dA, long ldA, long rows, long cols, const std::string &fileName)
{
  T matrix;

  std::ofstream file(fileName);

  if (file.is_open())
  {
    for (long i = 0; i < rows; i++)
    {
      for (long j = 0; j < cols; j++)
      {
        cudaMemcpy(&matrix, dA + i + j * ldA, sizeof(T), cudaMemcpyDeviceToHost);

        printf("%10.4f", matrix); 

        file << matrix;
        if ((cols - 1) != j)
        {
          file << ",";
        }
      }
      printf("\n");
      file << std::endl;
    }
    file.close();
    std::cout << std::endl << "Matrix written to " << fileName << std::endl;
  }
  else
  {
    std::cout << "Failed to open file: " << fileName << std::endl;
  }
}
