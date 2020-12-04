/*
 * reduce_shm.h
 *
 * Author: Ruben
 */

#pragma once

void reduceShm(int numBlocks, int blockSize, size_t shmSize, const int *sourceDev, int *destDev);
