/*
 * main.cpp
 *
 *  Created on: Dec 2, 2020
 *      Author: Rubén
 */
#include <assert.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>

#include "utils.h"
#include "reduce_shm.h"

// Standard CUDA API functions
#include <cuda_runtime_api.h>

int main(int argc, char **argv)
{
  const unsigned int COUNT = 4096 * 4096;
  std::unique_ptr<int[]> source(new int[COUNT]);

  // Fill source matrix with some arbitrary test values
  std::mt19937 rng;
  rng.seed(0);
  std::uniform_int_distribution<std::mt19937::result_type> dist(0, 9);

  for (int i = 0; i < COUNT; i++) {
    source[i] = dist(rng);
  }

  // Allocate and fill device memory
  int *source_dev, *dest_dev;
  size_t size = COUNT * sizeof(int);
  cudaCheckError(cudaMalloc((void**)&source_dev, size));
  cudaCheckError(
      cudaMemcpy(source_dev, source.get(), size, cudaMemcpyHostToDevice));

  // Run the kernel
  int BLOCK_SIZE = 128;
  int n_blocks = (COUNT + BLOCK_SIZE - 1) / (2 * BLOCK_SIZE);
  size_t shared_memory_size = BLOCK_SIZE * sizeof(int);

  cudaCheckError(cudaMalloc((void**)&dest_dev, n_blocks * sizeof(int)));
  reduce_shm(n_blocks, BLOCK_SIZE, shared_memory_size, source_dev, dest_dev);


  // Copy result back to the host
  int result;
  cudaCheckError(
      cudaMemcpy(&result, dest_dev, sizeof(result), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaFree(source_dev));
  cudaCheckError(cudaFree(dest_dev));

  // Compare with reference implementation
  int result_reference = std::accumulate(source.get(), source.get() + COUNT, 0);
  std::cout << "Sum of " << COUNT << " elements: " << result << "\n";
  assert(result_reference == result);

  return 0;
}




