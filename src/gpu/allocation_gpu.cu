/**
 * @file allocation_gpu.cu
 * @brief Source file defining methods regarding memory allocation on device.
 * @author Jan Warchocki
 * @date 10 March 2022
 */

#include "allocation_gpu.cuh"
#include "assert.cuh"
#include "verify.cuh"
#include <exceptions/unexpected_cuda_call_exception.h>

#ifdef __CUDA__

float* allocate1DArrayDevice(size_t n) {
    float* allocated;
    GPU_CHECK_ERROR(cudaMalloc(&allocated, n * sizeof(float)));
    return allocated;
}

void copy1DFromDeviceToDevice(float* oldLoc, float* newLoc, size_t n) {
    GPU_CHECK_ERROR(cudaMemcpy(newLoc, oldLoc, n * sizeof(float), cudaMemcpyDeviceToDevice));
}

void copy1DFromHostToDevice(float* host, float* device, size_t n) {
    GPU_CHECK_ERROR(cudaMemcpy(device, host, n * sizeof(float), cudaMemcpyHostToDevice));
}

void copy2DFromHostToDevice(float** host, float* device, size_t n, size_t m) {
    float* temp = allocate1DArray(n * m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            temp[i * m + j] = host[i][j];
        }
    }
    GPU_CHECK_ERROR(cudaMemcpy(device, temp, n * m * sizeof(float), cudaMemcpyHostToDevice));
    free(temp);
}

void free1DArrayDevice(float* device) {
    GPU_CHECK_ERROR(cudaFree(device));
}

void copy1DFromDeviceToHost(float* device, float* host, size_t n) {
    GPU_CHECK_ERROR(cudaMemcpy(host, device, n * sizeof(float), cudaMemcpyDeviceToHost));
}

void copy2DFromDeviceToHost(float* device, float** host, size_t n, size_t m) {
    float* temp = allocate1DArray(n * m);
    GPU_CHECK_ERROR(cudaMemcpy(temp, device, n * m * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            float val = temp[i * m + j];
            host[i][j] = val;
        }
    }

    free(temp);
}

float* copy1DArrayDevice(size_t n, float* old) {
    float* allocated = allocate1DArrayDevice(n);
    GPU_CHECK_ERROR(cudaMemcpy(allocated, old, n * sizeof(float), cudaMemcpyDeviceToDevice));
    return allocated;
}

#else

float* allocate1DArrayDevice(size_t n) {
    throw UnexpectedCUDACallException();
}

void copy1DFromDeviceToDevice(float* oldLoc, float* newLoc, size_t n) {
    throw UnexpectedCUDACallException();
}

void copy1DFromHostToDevice(float* host, float* device, size_t n) {
    throw UnexpectedCUDACallException();
}

void copy2DFromHostToDevice(float** host, float* device, size_t n, size_t m) {
    throw UnexpectedCUDACallException();
}

void free1DArrayDevice(float* device) {
    throw UnexpectedCUDACallException();
}

void copy1DFromDeviceToHost(float* device, float* host, size_t n) {
    throw UnexpectedCUDACallException();
}

void copy2DFromDeviceToHost(float* device, float** host, size_t n, size_t m) {
    throw UnexpectedCUDACallException();
}

float* copy1DArrayDevice(size_t n, float* old) {
    throw UnexpectedCUDACallException();
}

#endif