//
// Created by Jan Warchocki on 28/05/2022.
//

#include <utils/location_verifiers.h>
#include <exceptions/different_data_location_exception.h>
#include <gpu/assert.cuh>
#include <exceptions/unexpected_cuda_call_exception.h>
#include "../../../../include/activation.h"

#ifdef HAS_CUDA

__global__
void linearKernel(const DTYPE* vector, DTYPE* result, size_t n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    result[index] = vector[index];
}

__global__
void linearKernel(const DTYPE* matrix, DTYPE* result, size_t n, size_t m) {
    auto row = blockIdx.x;
    auto column = threadIdx.x;

    if (row >= n || column >= m) {
        return;
    }

    result[row * m + column] = matrix[row * m + column];
}

__global__
void linearDerivativeKernel(const DTYPE* vector, DTYPE* result, size_t n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    result[index] = 1;
}

__global__
void linearDerivativeKernel(const DTYPE* matrix, DTYPE* result, size_t n, size_t m) {
    auto row = blockIdx.x;
    auto column = threadIdx.x;

    if (row >= n || column >= m) {
        return;
    }

    result[row * m + column] = 1;
}

void LinearOnDeviceEvaluator::forward(const Vector& input, Vector& result) const {
    if (!allLocationsAreDevice({input.location, result.location})) {
        throw DifferentDataLocationException();
    }

    linearKernel<<<1, input.n>>>(input.data, result.data, input.n);
    gpuCheckError( cudaGetLastError() )
    gpuCheckError( cudaDeviceSynchronize() )
}

void LinearOnDeviceEvaluator::forward(const Matrix& input, Matrix& result) const {
    if (!allLocationsAreDevice({input.location, result.location})) {
        throw DifferentDataLocationException();
    }

    linearKernel<<<input.n, input.m>>>(input.data, result.data, input.n, input.m);
    gpuCheckError( cudaGetLastError() )
    gpuCheckError( cudaDeviceSynchronize() )
}

void LinearOnDeviceEvaluator::computeDerivatives(const Vector& output, Vector& result) const {
    if (!allLocationsAreDevice({output.location, result.location})) {
        throw DifferentDataLocationException();
    }

    linearDerivativeKernel<<<1, output.n>>>(output.data, result.data, output.n);
    gpuCheckError( cudaGetLastError() )
    gpuCheckError( cudaDeviceSynchronize() )
}

void LinearOnDeviceEvaluator::computeDerivatives(const Matrix& output, Matrix& result) const {
    if (!allLocationsAreDevice({output.location, result.location})) {
        throw DifferentDataLocationException();
    }

    linearDerivativeKernel<<<output.n, output.m>>>(output.data, result.data, output.n, output.m);
    gpuCheckError( cudaGetLastError() )
    gpuCheckError( cudaDeviceSynchronize() )
}

LinearOnDeviceEvaluator::~LinearOnDeviceEvaluator() = default;


#else

void LinearOnDeviceEvaluator::forward(const Vector& input, Vector& result) const {
    throw UnexpectedCUDACallException();
}

void LinearOnDeviceEvaluator::forward(const Matrix& input, Matrix& result) const {
    throw UnexpectedCUDACallException();
}

void LinearOnDeviceEvaluator::computeDerivatives(const Vector& output, Vector& result) const {
    throw UnexpectedCUDACallException();
}

void LinearOnDeviceEvaluator::computeDerivatives(const Matrix& output, Matrix& result) const {
    throw UnexpectedCUDACallException();
}

LinearOnDeviceEvaluator::~LinearOnDeviceEvaluator() = default;

#endif
