//
// Created by Jan Warchocki on 14/03/2022.
//

#include "gpu/allocation_gpu.cuh"
#include "gpu/assert.cuh"
#include "vector_operations_on_device.cuh"
#include "verify.cuh"
#include <exceptions/unexpected_cuda_call_exception.h>

#ifdef HAS_CUDA

//NOLINTBEGIN(readability-static-accessed-through-instance)

__global__ void addVectorsKernel(const DTYPE* v1, const DTYPE* v2, DTYPE* result, size_t n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    result[index] = v1[index] + v2[index];
}

__global__ void subtractVectorsKernel(const DTYPE* v1, const DTYPE* v2, DTYPE* result, size_t n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    result[index] = v1[index] - v2[index];
}

__global__ void multiplyVectorKernel(const DTYPE* v1, DTYPE constant, DTYPE* result, size_t n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    result[index] = v1[index] * constant;
}

//NOLINTEND(readability-static-accessed-through-instance)

void addVectorsOnDevice(const Vector& v1, const Vector& v2, Vector& result) {
    addVectorsKernel<<<1, v1.n>>>(v1.data, v2.data, result.data, v1.n);
    GPU_CHECK_ERROR(cudaGetLastError());
    GPU_CHECK_ERROR(cudaDeviceSynchronize());
}

void subtractVectorsOnDevice(const Vector& v1, const Vector& v2, Vector& result) {
    subtractVectorsKernel<<<1, v1.n>>>(v1.data, v2.data, result.data, v1.n);
    GPU_CHECK_ERROR(cudaGetLastError());
    GPU_CHECK_ERROR(cudaDeviceSynchronize());
}

void multiplyVectorOnDevice(const Vector& v1, DTYPE constant, Vector& result) {
    multiplyVectorKernel<<<1, v1.n>>>(v1.data, constant, result.data, v1.n);
    GPU_CHECK_ERROR(cudaGetLastError());
    GPU_CHECK_ERROR(cudaDeviceSynchronize());
}

#else

void addVectorsOnDevice(const Vector& v1, const Vector& v2, Vector& result) {
    throw UnexpectedCUDACallException();
}

void subtractVectorsOnDevice(const Vector& v1, const Vector& v2, Vector& result) {
    throw UnexpectedCUDACallException();
}

void multiplyVectorOnDevice(const Vector& v1, DTYPE constant, Vector& result) {
    throw UnexpectedCUDACallException();
}

#endif
