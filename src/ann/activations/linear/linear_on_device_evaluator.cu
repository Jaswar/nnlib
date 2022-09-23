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
 * @brief Kernel function for applying the activation function on a matrix of data.
 *
 * The function assumes the data samples are row aligned in the matrix.
 *
 * @param matrix The matrix on which to apply the activation function.
 * @param result The matrix where the result should be stored.
 * @param n The number of rows of the matrix.
 * @param m The number of columns of the matrix.
 */
__global__ void linearKernel(const DTYPE* input, DTYPE* result, size_t size) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size) {
        return;
    }

    result[index] = input[index];
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
__global__ void linearDerivativeKernel(const DTYPE* output, DTYPE* result, size_t size) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size) {
        return;
    }

    result[index] = 1;
}

// NOLINTEND(readability-static-accessed-through-instance)

void LinearOnDeviceEvaluator::forward(const Tensor& input, Tensor& result) const {
    if (!allLocationsAreDevice({input.location, result.location})) {
        throw DifferentDataLocationException();
    }

    auto grid = input.size / input.session.threadsPerBlock + 1;
    auto block = input.session.threadsPerBlock;
    linearKernel<<<grid, block>>>(input.host, result.host, input.size);
    GPU_CHECK_ERROR(cudaGetLastError());
}

void LinearOnDeviceEvaluator::computeDerivatives(const Tensor& output, Tensor& result) const {
    if (!allLocationsAreDevice({output.location, result.location})) {
        throw DifferentDataLocationException();
    }

    auto grid = output.size / output.session.threadsPerBlock + 1;
    auto block = output.session.threadsPerBlock;
    linearDerivativeKernel<<<grid, block>>>(output.host, result.host, output.size);
    GPU_CHECK_ERROR(cudaGetLastError());
}

LinearOnDeviceEvaluator::~LinearOnDeviceEvaluator() = default;


#else

void LinearOnDeviceEvaluator::forward(const Tensor& input, Tensor& result) const {
    throw UnexpectedCUDACallException();
}

void LinearOnDeviceEvaluator::computeDerivatives(const Tensor& output, Tensor& result) const {
    throw UnexpectedCUDACallException();
}

LinearOnDeviceEvaluator::~LinearOnDeviceEvaluator() = default;

#endif
