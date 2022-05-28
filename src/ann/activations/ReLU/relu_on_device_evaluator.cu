//
// Created by Jan Warchocki on 28/05/2022.
//

#include <activation.h>
#include <exceptions/unexpected_cuda_call_exception.h>
#include <utils/location_verifiers.h>
#include <exceptions/different_data_location_exception.h>
#include <gpu/assert.cuh>

#ifdef HAS_CUDA

__global__
void ReLUKernel(const DTYPE* vector, DTYPE* result, size_t n) {
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

__global__
void ReLUKernel(const DTYPE* matrix, DTYPE* result, size_t n, size_t m) {
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

__global__
void ReLUDerivativeKernel(const DTYPE* vector, DTYPE* result, size_t n) {
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

__global__
void ReLUDerivativeKernel(const DTYPE* matrix, DTYPE* result, size_t n, size_t m) {
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

void ReLUOnDeviceEvaluator::forward(const Vector& input, Vector& result) const {
    if (!allLocationsAreDevice({input.location, result.location})) {
        throw DifferentDataLocationException();
    }

    ReLUKernel<<<1, input.n>>>(input.data, result.data, input.n);
    gpuCheckError( cudaGetLastError() )
    gpuCheckError( cudaDeviceSynchronize() )
}

void ReLUOnDeviceEvaluator::forward(const Matrix& input, Matrix& result) const {
    if (!allLocationsAreDevice({input.location, result.location})) {
        throw DifferentDataLocationException();
    }

    ReLUKernel<<<input.n, input.m>>>(input.data, result.data, input.n, input.m);
    gpuCheckError( cudaGetLastError() )
    gpuCheckError( cudaDeviceSynchronize() )
}

void ReLUOnDeviceEvaluator::computeDerivatives(const Vector& output, Vector& result) const {
    if (!allLocationsAreDevice({output.location, result.location})) {
        throw DifferentDataLocationException();
    }

    ReLUDerivativeKernel<<<1, output.n>>>(output.data, result.data, output.n);
    gpuCheckError( cudaGetLastError() )
    gpuCheckError( cudaDeviceSynchronize() )
}

void ReLUOnDeviceEvaluator::computeDerivatives(const Matrix& output, Matrix& result) const {
    if (!allLocationsAreDevice({output.location, result.location})) {
        throw DifferentDataLocationException();
    }

    ReLUDerivativeKernel<<<output.n, output.m>>>(output.data, result.data, output.n, output.m);
    gpuCheckError( cudaGetLastError() )
    gpuCheckError( cudaDeviceSynchronize() )
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