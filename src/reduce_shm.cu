/*
 * reduce_shm.cu
 *
 * Author: Rubén
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

__device__ unsigned int blocksFinished = 0;
// Wait for all blocks in the grid to execute this function.
// Returns true for thread 0 of the last block, false for all
// other threads.
__device__ bool waitForAllBlocks()
{
	// Wait until global write is visible to all other blocks
	__threadfence();

	// Wait for all blocks to finish by atomically incrementing a counter
	bool isLast = false;
	if (threadIdx.x == 0) {
		unsigned int ticket = atomicInc(&blocksFinished, gridDim.x);
		isLast = (ticket == gridDim.x - 1);
	}
	if (isLast) {
		blocksFinished = 0;
	}
	return isLast;
}

__device__ int reduceBlock(const int *source, int sharedData[],
		cooperative_groups::thread_block block)
{
	unsigned int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;
	auto tid = threadIdx.x;

	// Add two elements into shared memory
	sharedData[tid] = source[index] + source[index + blockDim.x];

	cooperative_groups::sync(block);

	// When shared memory block is filled, reduce within that block.
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		int index = 2 * stride * tid;
		if (index < blockDim.x) {
			sharedData[index] += sharedData[index + stride];
		}
		cooperative_groups::sync(block);
	}

	return sharedData[0];
}

// Sum the source array. The dest array must have one element per block --
// the first element will contain the final result, and the rest are used for
// temporary storage.
__global__ void reduce(const int *source, int *dest)
{
	extern __shared__ int sharedData[];

	int block_result =
			reduceBlock(source, sharedData, cooperative_groups::this_thread_block());

	// The last thread of each block writes the block result into global memory
	if (threadIdx.x == 0) {
		dest[blockIdx.x] = block_result;
	}

	bool isLast = waitForAllBlocks();

	// All blocks have passed the threadfence, so all writes are visible to all
	// blocks. Now we can use one thread to sum the results from each block.
	if (isLast) {
		int sum = 0;
		for (int i = 0; i < gridDim.x; i++) {
			sum += dest[i];
		}
		// Final sum goes in dest[0]
		dest[0] = sum;
	}
}

void reduceShm(int numBlocks, int blockSize, size_t shmSize, const int *sourceDev, int *destDev)
{
	KernelTimer t;
	reduce<<<numBlocks, blockSize, shmSize>>>(sourceDev, destDev);
}
