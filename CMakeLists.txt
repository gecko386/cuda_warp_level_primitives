cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(shm_vs_warp LANGUAGES CXX CUDA)
find_package(CUDA 11)

include_directories(/usr/local/cuda/targets/x86_64-linux/include)
link_directories(/usr/local/cuda/targets/x86_64-linux/lib)

add_executable(shm_vs_warp  src/utils.cu
                            src/reduce_shm.cu
                            src/main.cpp)

target_compile_features(shm_vs_warp PUBLIC cxx_std_11)
 

