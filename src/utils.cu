/*
 * utils.cu
 *
 * Author: Ruben
 */

#include <iostream>
#include <cuda_runtime_api.h>
#include "utils.h"

KernelTimer::KernelTimer()
{
	cudaCheckError(cudaDeviceSynchronize());
	start = std::chrono::steady_clock::now();
}

KernelTimer::~KernelTimer()
{
	cudaCheckError(cudaDeviceSynchronize());
	auto end = std::chrono::steady_clock::now();
	auto elapsed =
			std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "kernel ran in " << elapsed << " ms\n";
}
