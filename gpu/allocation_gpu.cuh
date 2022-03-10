//
// Created by Jan Warchocki on 10/03/2022.
//

#ifndef NNLIB_ALLOCATION_GPU_CUH
#define NNLIB_ALLOCATION_GPU_CUH

#include "../utils/allocation.h"

DTYPE* allocate1DArrayDevice(int n);

void copy1DFromHostToDevice(DTYPE* host, DTYPE* device, int n);

void copy2DFromHostToDevice(DTYPE** host, DTYPE* device, int n, int m);

void copy1DFromDeviceToHost(DTYPE* device, DTYPE* host, int n);

void copy2DFromDeviceToHost(DTYPE* device, DTYPE** host, int n, int m);

void free1DArrayDevice(DTYPE* device);

#endif //NNLIB_ALLOCATION_GPU_CUH
