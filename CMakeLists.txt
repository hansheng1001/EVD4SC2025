cmake_minimum_required(VERSION 3.12)


set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb -DDEBUG")
set(CMAKE_C_FLAGS_DEBUG "$ENV{CFLAGS} -O0 -Wall -g -ggdb -DDEBUG")


message("${CMAKE_CXX_FLAGS_DEBUG}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CUDA_ARCHITECTURES 80)

#H100 90
#2880Ti 75
#4090 86
# A800 80

# set(CMAKE_CUDA_ARCHITECTURES 86) # for 4090

set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.5/bin/nvcc)
project(myqrcode LANGUAGES CXX CUDA)



add_subdirectory(src)



