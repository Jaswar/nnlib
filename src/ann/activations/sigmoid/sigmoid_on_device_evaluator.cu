/**
 * @file sigmoid_on_device_evaluator.cu
 * @brief Source file defining methods of the SigmoidOnDeviceEvaluator class.
 *
 * This also includes the definitions of GPU kernel functions that are used for `forward` and `computeDerivatives`
 * methods.
 *
 * @author Jan Warchocki
 * @date 28 May 2022
 */

#include "../../../../include/activation.h"
#include <cmath>
#include <exceptions/different_data_location_exception.h>
#include <exceptions/unexpected_cuda_call_exception.h>
#include <gpu/assert.cuh>
#include <utils/location_verifiers.h>

#ifdef HAS_CUDA

/**
 * @brief Kernel function to compute the output of the sigmoid function given the input.
 *
 * @param x The input to the sigmoid function.
 * @return The output of the sigmoid function.
 */
__device__ DTYPE fSigmoidKernel(DTYPE x) {
    return 1 / (1 + expf(-x));
}

// NOLINTBEGIN(readability-static-accessed-through-instance)

/** @copydoc linear_on_device_evaluator.cu::linearKernel(const DTYPE *vector, DTYPE *result, size_t n) */
__global__ void sigmoidKernel(DTYPE* vector, DTYPE* result, size_t n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    result[index] = fSigmoidKernel(vector[index]);
}

/** @copydoc linear_on_device_evaluator.cu::linearKernel(const DTYPE *matrix, DTYPE *result, size_t n, size_t m) */
__global__ void sigmoidKernel(DTYPE* matrix, DTYPE* result, size_t n, size_t m) {
    auto row = blockIdx.x;
    auto column = threadIdx.x;

    if (row >= n || column >= m) {
        return;
    }

    result[row * m + column] = fSigmoidKernel(matrix[row * m + column]);
}

/** @copydoc linear_on_device_evaluator.cu::linearDerivativeKernel(const DTYPE *vector, DTYPE *result, size_t n) */
__global__ void sigmoidDerivativeKernel(DTYPE* vector, DTYPE* result, size_t n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    result[index] = fSigmoidKernel(vector[index]) * (1 - fSigmoidKernel(vector[index]));
}

/** @copydoc linear_on_device_evaluator.cu::linearDerivativeKernel(const DTYPE *matrix, DTYPE *result, size_t n, size_t m) */
__global__ void sigmoidDerivativeKernel(DTYPE* matrix, DTYPE* result, size_t n, size_t m) {
    auto row = blockIdx.x;
    auto column = threadIdx.x;

    if (row >= n || column >= m) {
        return;
    }

    result[row * m + column] =
            fSigmoidKernel(matrix[row * m + column]) * (1 - fSigmoidKernel(matrix[row * m + column]));
}

// NOLINTEND(readability-static-accessed-through-instance)

void SigmoidOnDeviceEvaluator::forward(const Vector& input, Vector& result) const {
    if (!allLocationsAreDevice({input.location, result.location})) {
        throw DifferentDataLocationException();
    }

    sigmoidKernel<<<1, input.n>>>(input.data, result.data, input.n);
    GPU_CHECK_ERROR(cudaGetLastError());
}

void SigmoidOnDeviceEvaluator::forward(const Matrix& input, Matrix& result) const {
    if (!allLocationsAreDevice({input.location, result.location})) {
        throw DifferentDataLocationException();
    }

    sigmoidKernel<<<input.n, input.m>>>(input.data, result.data, input.n, input.m);
    GPU_CHECK_ERROR(cudaGetLastError());
}

void SigmoidOnDeviceEvaluator::computeDerivatives(const Vector& output, Vector& result) const {
    if (!allLocationsAreDevice({output.location, result.location})) {
        throw DifferentDataLocationException();
    }

    sigmoidDerivativeKernel<<<1, output.n>>>(output.data, result.data, output.n);
    GPU_CHECK_ERROR(cudaGetLastError());
}

void SigmoidOnDeviceEvaluator::computeDerivatives(const Matrix& output, Matrix& result) const {
    if (!allLocationsAreDevice({output.location, result.location})) {
        throw DifferentDataLocationException();
    }

    sigmoidDerivativeKernel<<<output.n, output.m>>>(output.data, result.data, output.n, output.m);
    GPU_CHECK_ERROR(cudaGetLastError());
}

SigmoidOnDeviceEvaluator::~SigmoidOnDeviceEvaluator() = default;

#else

void SigmoidOnDeviceEvaluator::forward(const Vector& input, Vector& result) const {
    throw UnexpectedCUDACallException();
}

void SigmoidOnDeviceEvaluator::forward(const Matrix& input, Matrix& result) const {
    throw UnexpectedCUDACallException();
}

void SigmoidOnDeviceEvaluator::computeDerivatives(const Vector& output, Vector& result) const {
    throw UnexpectedCUDACallException();
}

void SigmoidOnDeviceEvaluator::computeDerivatives(const Matrix& output, Matrix& result) const {
    throw UnexpectedCUDACallException();
}

SigmoidOnDeviceEvaluator::~SigmoidOnDeviceEvaluator() = default;

#endif