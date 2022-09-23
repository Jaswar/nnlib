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

/** @copydoc linear_on_device_evaluator.cu::linearKernel(const DTYPE *matrix, DTYPE *result, size_t n, size_t m) */
__global__ void sigmoidKernel(DTYPE* input, DTYPE* result, size_t size) {
    auto index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index >= size) {
        return;
    }

    result[index] = fSigmoidKernel(input[index]);
}

/** @copydoc linear_on_device_evaluator.cu::linearDerivativeKernel(const DTYPE *matrix, DTYPE *result, size_t n, size_t m) */
__global__ void sigmoidDerivativeKernel(DTYPE* output, DTYPE* result, size_t size) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size) {
        return;
    }

    result[index] = fSigmoidKernel(output[index]) * (1 - fSigmoidKernel(output[index]));
}

// NOLINTEND(readability-static-accessed-through-instance)

void SigmoidOnDeviceEvaluator::forward(const Tensor& input, Tensor& result) const {
    if (!allLocationsAreDevice({input.location, result.location})) {
        throw DifferentDataLocationException();
    }

    auto grid = input.size / input.session.threadsPerBlock + 1;
    auto block = input.session.threadsPerBlock;
    sigmoidKernel<<<grid, block>>>(input.device, result.device, input.size);
    GPU_CHECK_ERROR(cudaGetLastError());
}

void SigmoidOnDeviceEvaluator::computeDerivatives(const Tensor& output, Tensor& result) const {
    if (!allLocationsAreDevice({output.location, result.location})) {
        throw DifferentDataLocationException();
    }

    auto grid = output.size / output.session.threadsPerBlock + 1;
    auto block = output.session.threadsPerBlock;
    sigmoidDerivativeKernel<<<grid, block>>>(output.device, result.device, output.size);
    GPU_CHECK_ERROR(cudaGetLastError());
}

SigmoidOnDeviceEvaluator::~SigmoidOnDeviceEvaluator() = default;

#else

void SigmoidOnDeviceEvaluator::forward(const Tensor& input, Tensor& result) const {
    throw UnexpectedCUDACallException();
}

void SigmoidOnDeviceEvaluator::computeDerivatives(const Tensor& output, Tensor& result) const {
    throw UnexpectedCUDACallException();
}

SigmoidOnDeviceEvaluator::~SigmoidOnDeviceEvaluator() = default;

#endif