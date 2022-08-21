/**
 * @file vector_operations_on_device.cu
 * @brief Source file defining vector operations that happen on device.
 * @author Jan Warchocki
 * @date 14 March 2022
 */

#include "gpu/allocation_gpu.cuh"
#include "gpu/assert.cuh"
#include "vector_operations_on_device.cuh"
#include "verify.cuh"
#include <exceptions/unexpected_cuda_call_exception.h>

#ifdef HAS_CUDA

// NOLINTBEGIN(readability-static-accessed-through-instance)

/**
 * @brief Kernel method to add two vectors together.
 *
 * @param v1 The data of the first vector.
 * @param v2 The data of the second vector.
 * @param result Where the result of the addition should be stored.
 * @param n The size of the vectors.
 */
__global__ void addVectorsKernel(const DTYPE* v1, const DTYPE* v2, DTYPE* result, size_t n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    result[index] = v1[index] + v2[index];
}

/**
 * @brief Kernel method to subtract one vector from another.
 *
 * @param v1 The data of the vector to subtract from.
 * @param v2 The data of the vector that should be subtracted.
 * @param result Where the result of the subtraction should be stored.
 * @param n The size of the vectors.
 */
__global__ void subtractVectorsKernel(const DTYPE* v1, const DTYPE* v2, DTYPE* result, size_t n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    result[index] = v1[index] - v2[index];
}

/**
 * @brief Kernel method to multiply a vector with a constant.
 *
 * @param v1 The data of the vector to multiply.
 * @param constant The constant by which to multiply the vector.
 * @param result Where the result of the multiplication should be saved.
 * @param n The size of the vectors.
 */
__global__ void multiplyVectorKernel(const DTYPE* v1, DTYPE constant, DTYPE* result, size_t n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    result[index] = v1[index] * constant;
}

// NOLINTEND(readability-static-accessed-through-instance)

void addVectorsOnDevice(const Vector& v1, const Vector& v2, Vector& result) {
    addVectorsKernel<<<1, v1.n>>>(v1.data, v2.data, result.data, v1.n);
    GPU_CHECK_ERROR(cudaGetLastError());
}

void subtractVectorsOnDevice(const Vector& v1, const Vector& v2, Vector& result) {
    subtractVectorsKernel<<<1, v1.n>>>(v1.data, v2.data, result.data, v1.n);
    GPU_CHECK_ERROR(cudaGetLastError());
}

void multiplyVectorOnDevice(const Vector& v1, DTYPE constant, Vector& result) {
    multiplyVectorKernel<<<1, v1.n>>>(v1.data, constant, result.data, v1.n);
    GPU_CHECK_ERROR(cudaGetLastError());
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
