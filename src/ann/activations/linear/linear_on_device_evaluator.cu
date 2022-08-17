/**
 * @file linear_on_device_evaluator.cu
 * @brief Source file defining methods of the LinearOnDeviceEvaluator class.
 *
 * This also includes the definitions of GPU kernel functions that are used for `forward` and `computeDerivatives`
 * methods.
 *
 * @author Jan Warchocki
 * @date 28 May 2022
 */

#include "../../../../include/activation.h"
#include <exceptions/different_data_location_exception.h>
#include <exceptions/unexpected_cuda_call_exception.h>
#include <gpu/assert.cuh>
#include <utils/location_verifiers.h>

#ifdef HAS_CUDA

// NOLINTBEGIN(readability-static-accessed-through-instance)

/**
 * @brief Kernel function for applying the activation function on a vector of data.
 *
 * @param vector The vector on which to apply the activation function.
 * @param result The vector where the result should be stored.
 * @param n The size of the vector.
 */
__global__ void linearKernel(const DTYPE* vector, DTYPE* result, size_t n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    result[index] = vector[index];
}

/**
 * @brief Kernel function for applying the activation function on a matrix of data.
 *
 * The function assumes the data samples are row aligned in the matrix.
 *
 * @param matrix The matrix on which to apply the activation function.
 * @param result The matrix where the result should be stored.
 * @param n The number of rows of the matrix.
 * @param m The number of columns of the matrix.
 */
__global__ void linearKernel(const DTYPE* matrix, DTYPE* result, size_t n, size_t m) {
    auto row = blockIdx.x;
    auto column = threadIdx.x;

    if (row >= n || column >= m) {
        return;
    }

    result[row * m + column] = matrix[row * m + column];
}

/**
 * @brief Kernel function for computing derivatives of a vector of data.
 *
 * @param vector The vector whose derivatives to compute.
 * @param result Where the derivatives should be stored.
 * @param n The size of the vector.
 */
__global__ void linearDerivativeKernel(const DTYPE* vector, DTYPE* result, size_t n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    result[index] = 1;
}

/**
 * @brief Kernel function for computing derivatives of a matrix of data.
 *
 * The function assumes the data samples are row aligned in the matrix.
 *
 * @param matrix The matrix whose derivatives to compute.
 * @param result Where the derivatives should be stored.
 * @param n The number of rows of the matrix.
 * @param m The number of columns of the matrix.
 */
__global__ void linearDerivativeKernel(const DTYPE* matrix, DTYPE* result, size_t n, size_t m) {
    auto row = blockIdx.x;
    auto column = threadIdx.x;

    if (row >= n || column >= m) {
        return;
    }

    result[row * m + column] = 1;
}

// NOLINTEND(readability-static-accessed-through-instance)

void LinearOnDeviceEvaluator::forward(const Vector& input, Vector& result) const {
    if (!allLocationsAreDevice({input.location, result.location})) {
        throw DifferentDataLocationException();
    }

    linearKernel<<<1, input.n>>>(input.data, result.data, input.n);
    GPU_CHECK_ERROR(cudaGetLastError());
}

void LinearOnDeviceEvaluator::forward(const Matrix& input, Matrix& result) const {
    if (!allLocationsAreDevice({input.location, result.location})) {
        throw DifferentDataLocationException();
    }

    linearKernel<<<input.n, input.m>>>(input.data, result.data, input.n, input.m);
    GPU_CHECK_ERROR(cudaGetLastError());
}

void LinearOnDeviceEvaluator::computeDerivatives(const Vector& output, Vector& result) const {
    if (!allLocationsAreDevice({output.location, result.location})) {
        throw DifferentDataLocationException();
    }

    linearDerivativeKernel<<<1, output.n>>>(output.data, result.data, output.n);
    GPU_CHECK_ERROR(cudaGetLastError());
}

void LinearOnDeviceEvaluator::computeDerivatives(const Matrix& output, Matrix& result) const {
    if (!allLocationsAreDevice({output.location, result.location})) {
        throw DifferentDataLocationException();
    }

    linearDerivativeKernel<<<output.n, output.m>>>(output.data, result.data, output.n, output.m);
    GPU_CHECK_ERROR(cudaGetLastError());
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
