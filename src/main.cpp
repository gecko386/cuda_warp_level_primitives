/*
 * main.cpp
 *
 *  Created on: Dec 2, 2020
 *      Author: Rub��n
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
	const unsigned int count = 8192 * 8192;
	const int blockSize = 128;
	int numBlocks = (count + blockSize - 1) / (2 * blockSize);
	size_t shmSize = blockSize * sizeof(int);

	std::unique_ptr<int[]> source(new int[count]);

	// Fill source matrix with some arbitrary test values
	std::mt19937 rng;
	rng.seed(0);
	std::uniform_int_distribution<std::mt19937::result_type> dist(0, 9);

	for (int i = 0; i < count; i++) {
		source[i] = dist(rng);
	}

	// Allocate and fill device memory
	int *sourceDev, *destDev;
	size_t size = count * sizeof(int);
	cudaCheckError(cudaMalloc((void**)&sourceDev, size));
	cudaCheckError(
			cudaMemcpy(sourceDev, source.get(), size, cudaMemcpyHostToDevice));

	// Run the kernel

	cudaCheckError(cudaMalloc((void**)&destDev, numBlocks * sizeof(int)));
	reduceShm(numBlocks, blockSize, shmSize, sourceDev, destDev);


	// Copy result back to the host
	int result;
	cudaCheckError(
			cudaMemcpy(&result, destDev, sizeof(result), cudaMemcpyDeviceToHost));
	cudaCheckError(cudaFree(sourceDev));
	cudaCheckError(cudaFree(destDev));

	// Compare with reference implementation
	int result_reference = std::accumulate(source.get(), source.get() + count, 0);
	std::cout << "Sum of " << count << " elements: " << result << "\n";
	assert(result_reference == result);

	return 0;
}




