/*
 * utils.h
 *
 * Author: Ruben
 */

#pragma once

#include <chrono>

// Error checking macro
#define cudaCheckError(code)                                             \
		{                                                                      \
	if ((code) != cudaSuccess) {                                         \
		fprintf(stderr, "Cuda failure %s:%d: '%s' \n", __FILE__, __LINE__, \
				cudaGetErrorString(code));                                 \
	}                                                                    \
		}

class KernelTimer
{
public:
	KernelTimer();
	~KernelTimer();

private:
	std::chrono::time_point<std::chrono::steady_clock> start;
};

