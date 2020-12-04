cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(shm_vs_warp LANGUAGES CXX CUDA)
enable_language(CUDA)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(CUDA_NVCC_FLAGS "-gencode arch=compute_50,code=sm_50;-rdc=true;-use_fast_math")

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


add_executable(shm_vs_warp  src/utils.cu
                            src/reduce_shm.cu
                            src/main.cpp)

target_compile_features(shm_vs_warp PUBLIC cxx_std_11)

 
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70")
