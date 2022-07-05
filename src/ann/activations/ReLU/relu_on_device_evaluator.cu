//
// Created by Jan Warchocki on 28/05/2022.
//

#include "../../../../include/activation.h"
#include <exceptions/different_data_location_exception.h>
#include <exceptions/unexpected_cuda_call_exception.h>
#include <gpu/assert.cuh>
#include <utils/location_verifiers.h>

#ifdef HAS_CUDA

//NOLINTBEGIN(readability-static-accessed-through-instance)

__global__ void reluKernel(const DTYPE* vector, DTYPE* result, size_t n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    if (vector[index] <= 0) {
        result[index] = 0;
    } else {
        result[index] = vector[index];
    }
}

__global__ void reluKernel(const DTYPE* matrix, DTYPE* result, size_t n, size_t m) {
    auto row = blockIdx.x;
    auto column = threadIdx.x;

    if (row >= n || column >= m) {
        return;
    }

    if (matrix[row * m + column] <= 0) {
        result[row * m + column] = 0;
    } else {
        result[row * m + column] = matrix[row * m + column];
    }
}

__global__ void reluDerivativeKernel(const DTYPE* vector, DTYPE* result, size_t n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    if (vector[index] <= 0) {
        result[index] = 0;
    } else {
        result[index] = 1;
    }
}

__global__ void reluDerivativeKernel(const DTYPE* matrix, DTYPE* result, size_t n, size_t m) {
    auto row = blockIdx.x;
    auto column = threadIdx.x;

    if (row >= n || column >= m) {
        return;
    }

    if (matrix[row * m + column] <= 0) {
        result[row * m + column] = 0;
    } else {
        result[row * m + column] = 1;
    }
}

//NOLINTEND(readability-static-accessed-through-instance)

void ReLUOnDeviceEvaluator::forward(const Vector& input, Vector& result) const {
    if (!allLocationsAreDevice({input.location, result.location})) {
        throw DifferentDataLocationException();
    }

    reluKernel<<<1, input.n>>>(input.data, result.data, input.n);
    GPU_CHECK_ERROR(cudaGetLastError());
    GPU_CHECK_ERROR(cudaDeviceSynchronize());
}

void ReLUOnDeviceEvaluator::forward(const Matrix& input, Matrix& result) const {
    if (!allLocationsAreDevice({input.location, result.location})) {
        throw DifferentDataLocationException();
    }

    reluKernel<<<input.n, input.m>>>(input.data, result.data, input.n, input.m);
    GPU_CHECK_ERROR(cudaGetLastError());
    GPU_CHECK_ERROR(cudaDeviceSynchronize());
}

void ReLUOnDeviceEvaluator::computeDerivatives(const Vector& output, Vector& result) const {
    if (!allLocationsAreDevice({output.location, result.location})) {
        throw DifferentDataLocationException();
    }

    reluDerivativeKernel<<<1, output.n>>>(output.data, result.data, output.n);
    GPU_CHECK_ERROR(cudaGetLastError());
    GPU_CHECK_ERROR(cudaDeviceSynchronize());
}

void ReLUOnDeviceEvaluator::computeDerivatives(const Matrix& output, Matrix& result) const {
    if (!allLocationsAreDevice({output.location, result.location})) {
        throw DifferentDataLocationException();
    }

    reluDerivativeKernel<<<output.n, output.m>>>(output.data, result.data, output.n, output.m);
    GPU_CHECK_ERROR(cudaGetLastError());
    GPU_CHECK_ERROR(cudaDeviceSynchronize());
}

ReLUOnDeviceEvaluator::~ReLUOnDeviceEvaluator() = default;

#else

void ReLUOnDeviceEvaluator::forward(const Vector& input, Vector& result) const {
    throw UnexpectedCUDACallException();
}

void ReLUOnDeviceEvaluator::forward(const Matrix& input, Matrix& result) const {
    throw UnexpectedCUDACallException();
}

void ReLUOnDeviceEvaluator::computeDerivatives(const Vector& output, Vector& result) const {
    throw UnexpectedCUDACallException();
}

void ReLUOnDeviceEvaluator::computeDerivatives(const Matrix& output, Matrix& result) const {
    throw UnexpectedCUDACallException();
}

ReLUOnDeviceEvaluator::~ReLUOnDeviceEvaluator() = default;

#endif
