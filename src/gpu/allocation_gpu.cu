//
// Created by Jan Warchocki on 10/03/2022.
//

#include "allocation_gpu.cuh"
#include "assert.cuh"
#include "verify.cuh"
#include <exceptions/unexpected_cuda_call_exception.h>

#ifdef HAS_CUDA

DTYPE* allocate1DArrayDevice(size_t n) {
    DTYPE* allocated;
    gpuCheckError(cudaMalloc(&allocated, n * sizeof(DTYPE)));
    return allocated;
}

void copy1DFromDeviceToDevice(DTYPE* oldLoc, DTYPE* newLoc, size_t n) {
    gpuCheckError(cudaMemcpy(newLoc, oldLoc, n * sizeof(DTYPE), cudaMemcpyDeviceToDevice));
}

void copy1DFromHostToDevice(DTYPE* host, DTYPE* device, size_t n) {
    gpuCheckError(cudaMemcpy(device, host, n * sizeof(DTYPE), cudaMemcpyHostToDevice));
}

void copy2DFromHostToDevice(DTYPE** host, DTYPE* device, size_t n, size_t m) {
    DTYPE* temp = allocate1DArray(n * m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            temp[i * m + j] = host[i][j];
        }
    }
    gpuCheckError(cudaMemcpy(device, temp, n * m * sizeof(DTYPE), cudaMemcpyHostToDevice));
    free(temp);
}

void free1DArrayDevice(DTYPE* device) {
    gpuCheckError(cudaFree(device));
}

void copy1DFromDeviceToHost(DTYPE* device, DTYPE* host, size_t n) {
    gpuCheckError(cudaMemcpy(host, device, n * sizeof(DTYPE), cudaMemcpyDeviceToHost));
}

void copy2DFromDeviceToHost(DTYPE* device, DTYPE** host, size_t n, size_t m) {
    DTYPE* temp = allocate1DArray(n * m);
    gpuCheckError(cudaMemcpy(temp, device, n * m * sizeof(DTYPE), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            DTYPE val = temp[i * m + j];
            host[i][j] = val;
        }
    }

    free(temp);
}

DTYPE* copy1DArrayDevice(size_t n, DTYPE* old) {
    DTYPE* allocated = allocate1DArrayDevice(n);
    gpuCheckError(cudaMemcpy(allocated, old, n * sizeof(DTYPE), cudaMemcpyDeviceToDevice));
    return allocated;
}

#else

DTYPE* allocate1DArrayDevice(size_t n) {
    throw UnexpectedCUDACallException();
}

void copy1DFromDeviceToDevice(DTYPE* oldLoc, DTYPE* newLoc, size_t n) {
    throw UnexpectedCUDACallException();
}

void copy1DFromHostToDevice(DTYPE* host, DTYPE* device, size_t n) {
    throw UnexpectedCUDACallException();
}

void copy2DFromHostToDevice(DTYPE** host, DTYPE* device, size_t n, size_t m) {
    throw UnexpectedCUDACallException();
}

void free1DArrayDevice(DTYPE* device) {
    throw UnexpectedCUDACallException();
}

void copy1DFromDeviceToHost(DTYPE* device, DTYPE* host, size_t n) {
    throw UnexpectedCUDACallException();
}

void copy2DFromDeviceToHost(DTYPE* device, DTYPE** host, size_t n, size_t m) {
    throw UnexpectedCUDACallException();
}

DTYPE* copy1DArrayDevice(size_t n, DTYPE* old) {
    throw UnexpectedCUDACallException();
}

#endif