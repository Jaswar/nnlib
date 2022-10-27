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
 * @brief Kernel function for applying the activation function on a tensor of data.
 *
 * @param input The tensor on which to apply the activation function.
 * @param result The matrix where the result should be stored.
 * @param size The size of the tensor.
 */
__global__ void linearKernel(const float* input, float* result, size_t size) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size) {
        return;
    }

    result[index] = input[index];
}

/**
 * @brief Kernel function for computing derivatives of a tensor of data.
 *
 * @param output The tensor whose derivatives to compute.
 * @param result Where the derivatives should be stored.
 * @param size The size of the tensor.
 */
__global__ void linearDerivativeKernel(const float* output, float* result, size_t size) {
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
    linearKernel<<<grid, block>>>(input.data, result.data, input.size);
    GPU_CHECK_ERROR(cudaGetLastError());
}

void LinearOnDeviceEvaluator::computeDerivatives(const Tensor& output, Tensor& result) const {
    if (!allLocationsAreDevice({output.location, result.location})) {
        throw DifferentDataLocationException();
    }

    auto grid = output.size / output.session.threadsPerBlock + 1;
    auto block = output.session.threadsPerBlock;
    linearDerivativeKernel<<<grid, block>>>(output.data, result.data, output.size);
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
