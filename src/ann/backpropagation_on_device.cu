//
// Created by Jan Warchocki on 29/05/2022.
//

#include <exceptions/unexpected_cuda_call_exception.h>
#include "backpropagation.h"
#include "verify.cuh"
#include "../gpu/allocation_gpu.cuh"
#include "../gpu/assert.cuh"

#ifdef HAS_CUDA

__global__
void applyGradientsKernel(DTYPE* biases, DTYPE* weights, DTYPE* biasesGradients, DTYPE* weightsGradients,
                          size_t inSize, size_t outSize, size_t batchSize, DTYPE learningRate) {
    auto outIndex = blockIdx.x;
    auto inIndex = threadIdx.x;

    if (outIndex >= outSize || inIndex >= inSize) {
        return;
    }

    if (inIndex == 0) {
        biases[outIndex] -= learningRate * biasesGradients[outIndex] / (DTYPE) batchSize;
        biasesGradients[outIndex] = 0;
    }

    weights[inIndex * outSize + outIndex] -= learningRate * weightsGradients[inIndex * outSize + outIndex] / (DTYPE) batchSize;
    weightsGradients[inIndex * outSize + outIndex] = 0;
}

void applyGradientsOnDevice(Layer& layer, size_t batchSize, DTYPE learningRate) {
    applyGradientsKernel<<<layer.outSize, layer.inSize>>>(layer.biases.data, layer.weights.data,
                                                          layer.biasesGradients.data, layer.weightsGradients.data,
                                                          layer.inSize, layer.outSize, batchSize, learningRate);
    gpuCheckError( cudaGetLastError() )
    gpuCheckError( cudaDeviceSynchronize() )
}

#else

void applyGradientsOnDevice(Layer& layer, size_t batchSize, DTYPE learningRate) {
    throw UnexpectedCUDACallException();
}

#endif