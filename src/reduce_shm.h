/*
 * reduce_cu.h
 *
 *  Created on: Dec 2, 2020
 *      Author: Rubén
 */

#pragma once

void reduce_shm(int numBlocks, int blockSize, size_t shmSize, const int *source_dev, int *dest_dev);
