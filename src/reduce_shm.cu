/*
 * reduce_shm.cu
 *
 *  Created on: Dec 2, 2020
 *      Author: Rubén
 */

#include <assert.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>

// Standard CUDA API functions
#include <cuda_runtime_api.h>

// CUDA cooperative groups API
#include <cooperative_groups.h>

#include "utils.h"


__device__ unsigned int blocks_finished = 0;
// Wait for all blocks in the grid to execute this function.
// Returns true for thread 0 of the last block, false for all
// other threads.
__device__ bool wait_for_all_blocks()
{
  // Wait until global write is visible to all other blocks
  __threadfence();

  // Wait for all blocks to finish by atomically incrementing a counter
  bool is_last = false;
  if (threadIdx.x == 0) {
    unsigned int ticket = atomicInc(&blocks_finished, gridDim.x);
    is_last = (ticket == gridDim.x - 1);
  }
  if (is_last) {
    blocks_finished = 0;
  }
  return is_last;
}

__device__ int reduce_block(const int *source, int sdata[],
                            cooperative_groups::thread_block block)
{
  unsigned int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  auto tid = threadIdx.x;

  // Add two elements into shared memory
  sdata[tid] = source[index] + source[index + blockDim.x];

  cooperative_groups::sync(block);

  // When shared memory block is filled, reduce within that block.
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    int index = 2 * stride * tid;
    if (index < blockDim.x) {
      sdata[index] += sdata[index + stride];
    }
    cooperative_groups::sync(block);
  }

  return sdata[0];
}

// Sum the source array. The dest array must have one element per block --
// the first element will contain the final result, and the rest are used for
// temporary storage.
__global__ void reduce(const int *source, int *dest)
{
  extern __shared__ int sdata[];

  int block_result =
      reduce_block(source, sdata, cooperative_groups::this_thread_block());

  // The last thread of each block writes the block result into global memory
  if (threadIdx.x == 0) {
    dest[blockIdx.x] = block_result;
  }

  bool is_last = wait_for_all_blocks();

  // All blocks have passed the threadfence, so all writes are visible to all
  // blocks. Now we can use one thread to sum the results from each block.
  if (is_last) {
    int sum = 0;
    for (int i = 0; i < gridDim.x; i++) {
      sum += dest[i];
    }
    // Final sum goes in dest[0]
    dest[0] = sum;
  }
}

void reduce_shm(int numBlocks, int blockSize, size_t shmSize, const int *source_dev, int *dest_dev)
{
    KernelTimer t;
    reduce<<<numBlocks, blockSize, shmSize>>>(source_dev, dest_dev);
}
