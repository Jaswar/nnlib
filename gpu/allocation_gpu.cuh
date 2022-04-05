//
// Created by Jan Warchocki on 10/03/2022.
//

#ifndef NNLIB_ALLOCATION_GPU_CUH
#define NNLIB_ALLOCATION_GPU_CUH

#include "../utils/allocation.h"

DTYPE* allocate1DArrayDevice(size_t n);


void copy1DFromHostToDevice(DTYPE* host, DTYPE* device, size_t n);

void copy2DFromHostToDevice(DTYPE** host, DTYPE* device, size_t n, size_t m);

void copy1DFromDeviceToHost(DTYPE* device, DTYPE* host, size_t n);

void copy2DFromDeviceToHost(DTYPE* device, DTYPE** host, size_t n, size_t m);

DTYPE* copy1DArrayDevice(size_t n, DTYPE* old);

void free1DArrayDevice(DTYPE* device);

#endif //NNLIB_ALLOCATION_GPU_CUH
