/**
 * @file relu_on_device_evaluator.cu
 * @brief Source file defining methods of the ReLUOnDeviceEvaluator class.
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

/** @copydoc linear_on_device_evaluator.cu::linearKernel(const float *matrix, float *result, size_t n, size_t m) */
__global__ void reluKernel(const float* input, float* result, size_t size) {
    auto index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index >= size) {
        return;
    }

    if (input[index] <= 0) {
        result[index] = 0;
    } else {
        result[index] = input[index];
    }
}

/** @copydoc linear_on_device_evaluator.cu::linearDerivativeKernel(const float *matrix, float *result, size_t n, size_t m) */
__global__ void reluDerivativeKernel(const float* output, float* result, size_t size) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size) {
        return;
    }

    if (output[index] <= 0) {
        result[index] = 0;
    } else {
        result[index] = 1;
    }
}

// NOLINTEND(readability-static-accessed-through-instance)

void ReLUOnDeviceEvaluator::forward(const Tensor& input, Tensor& result) const {
    if (!allLocationsAreDevice({input.location, result.location})) {
        throw DifferentDataLocationException();
    }

    auto grid = input.size / input.session.threadsPerBlock + 1;
    auto block = input.session.threadsPerBlock;
    reluKernel<<<grid, block>>>(input.device, result.device, input.size);
    GPU_CHECK_ERROR(cudaGetLastError());
}

void ReLUOnDeviceEvaluator::computeDerivatives(const Tensor& output, Tensor& result) const {
    if (!allLocationsAreDevice({output.location, result.location})) {
        throw DifferentDataLocationException();
    }

    auto grid = output.size / output.session.threadsPerBlock + 1;
    auto block = output.session.threadsPerBlock;
    reluDerivativeKernel<<<grid, block>>>(output.device, result.device, output.size);
    GPU_CHECK_ERROR(cudaGetLastError());
}

ReLUOnDeviceEvaluator::~ReLUOnDeviceEvaluator() = default;

#else

void ReLUOnDeviceEvaluator::forward(const Tensor& input, Tensor& result) const {
    throw UnexpectedCUDACallException();
}

void ReLUOnDeviceEvaluator::computeDerivatives(const Tensor& output, Tensor& result) const {
    throw UnexpectedCUDACallException();
}

ReLUOnDeviceEvaluator::~ReLUOnDeviceEvaluator() = default;

#endif
