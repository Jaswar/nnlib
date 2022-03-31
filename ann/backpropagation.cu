//
// Created by Jan Warchocki on 10/03/2022.
//

#include "backpropagation.cuh"
#include "../gpu/verify.cuh"
#include "../gpu/allocation_gpu.cuh"
#include "../gpu/assert.cuh"

#ifdef HAS_CUDA

__global__
void computeGradientsDevice(DTYPE* biasesGradients, DTYPE* weightsGradients, const DTYPE* data, const DTYPE* derivatives,
                            const DTYPE* delta, const DTYPE* previousWeights, DTYPE* newDelta,
                            int inSize, int outSize, int deltaSize, int batchSize, bool isLastLayer) {
    auto outIndex = blockIdx.x;
    auto inIndex = threadIdx.x;

    if (outIndex >= outSize || inIndex >= inSize) {
        return;
    }

    for (int row = 0; row < batchSize; row++) {
        if (isLastLayer) {
            DTYPE coreGradient = delta[row * deltaSize + outIndex] * derivatives[row * outSize + outIndex];

            if (inIndex == 0) {
                newDelta[row * outSize + outIndex] = coreGradient;

                biasesGradients[outIndex] += coreGradient;
            }

            weightsGradients[inIndex * outSize + outIndex] += coreGradient * data[row * inSize + inIndex];
        } else {
            DTYPE coreGradient = 0;
            for (int j = 0; j < deltaSize; j++) {
                coreGradient += delta[row * deltaSize + j] * derivatives[row * outSize + outIndex] * previousWeights[outIndex * deltaSize + j];
            }

            if (inIndex == 0) {
                newDelta[row * outSize + outIndex] = coreGradient;

                biasesGradients[outIndex] += coreGradient;
            }

            weightsGradients[inIndex * outSize + outIndex] += coreGradient * data[row * inSize + inIndex];
        }
    }

}

void computeGradients(Layer& layer, const Matrix& delta, const Matrix& previousWeights,
                     int batchSize, bool isLastLayer) {
    computeGradientsDevice<<<layer.outSize, layer.inSize>>>(layer.biasesGradients.data, layer.weightsGradients.data,
                                                 layer.data->data, layer.derivatives.data, delta.data,
                                                 previousWeights.data, layer.newDelta.data,
                                                 layer.inSize, layer.outSize, delta.m, batchSize, isLastLayer);
    gpuCheckError( cudaDeviceSynchronize() )
}

__global__
void applyGradientsDevice(DTYPE* biases, DTYPE* weights, DTYPE* biasesGradients, DTYPE* weightsGradients,
                          int inSize, int outSize, int batchSize, DTYPE learningRate) {
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

void applyGradients(Layer& layer, int batchSize, DTYPE learningRate) {
    applyGradientsDevice<<<layer.outSize, layer.inSize>>>(layer.biases.data, layer.weights.data,
                                               layer.biasesGradients.data, layer.weightsGradients.data,
                                               layer.inSize, layer.outSize, batchSize, learningRate);
    gpuCheckError( cudaDeviceSynchronize() )
}

#else

Vector backpropagation(Layer& layer, const Vector& delta, const Matrix& previousWeights,
              bool isLastLayer, DTYPE learningRate) {
    const Vector& derivatives = layer.calculateDerivatives();
    if (!isLastLayer) {
        Vector newDelta = Vector(layer.outSize);
        for (int i = 0; i < layer.outSize; i++) {
            DTYPE coreGradient = 0;
            for (int j = 0; j < delta.n; j++) {
                coreGradient += delta[j] * derivatives[i] * previousWeights[j][i];
            }

            layer.biases[i] -= learningRate * coreGradient;

            for (int j = 0; j < layer.inSize; j++) {
                layer.weights[i][j] -= learningRate * coreGradient * layer.data[j];
            }

            newDelta[i] = coreGradient;
        }

        return newDelta;
    } else {
        Vector newDelta = delta;
        for (int i = 0; i < layer.outSize; i++) {
            DTYPE coreGradient = delta[i] * derivatives[i];

            layer.biases[i] -= learningRate * coreGradient;

            for (int j = 0; j < layer.inSize; j++) {
                layer.weights[i][j] -= learningRate * coreGradient * layer.data[j];
            }

            newDelta.data[i] = coreGradient;
        }
        return newDelta;
    }
}

#endif