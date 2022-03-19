//
// Created by Jan Warchocki on 10/03/2022.
//

#include "backpropagation.cuh"
#include "../gpu/verify.cuh"
#include "../gpu/allocation_gpu.cuh"
#include "../gpu/assert.cuh"

#ifdef HAS_CUDA

__global__
void performBackpropagation(DTYPE* biases, DTYPE* weights, DTYPE* data, DTYPE* derivatives,
                            DTYPE* delta, DTYPE* previousWeights, DTYPE* newDelta, int inSize, int outSize, int deltaSize,
                            DTYPE learningRate, bool isLastLayer) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= outSize) {
        return;
    }

    if (isLastLayer) {
        DTYPE coreGradient = delta[index] * derivatives[index];

        biases[index] -= learningRate * coreGradient;

        for (int j = 0; j < inSize; j++) {
            weights[index * inSize + j] -= learningRate * coreGradient * data[j];
        }

        newDelta[index] = coreGradient;
    } else {
        DTYPE coreGradient = 0;
        for (int j = 0; j < deltaSize; j++) {
            coreGradient += delta[j] * derivatives[index] * previousWeights[j * outSize + index];
        }

        biases[index] -= learningRate * coreGradient;

        for (int j = 0; j < inSize; j++) {
            weights[index * inSize + j] -= learningRate * coreGradient * data[j];
        }

        newDelta[index] = coreGradient;
    }
}

void backpropagation(Layer& layer, const Vector& delta, const Matrix& previousWeights,
              bool isLastLayer, DTYPE learningRate) {
    performBackpropagation<<<1, layer.outSize>>>(layer.biases.data, layer.weights.data,
                                                 layer.data->data, layer.derivatives.data, delta.data,
                                                 previousWeights.data, layer.newDelta.data,
                                                 layer.inSize, layer.outSize, delta.n, learningRate, isLastLayer);
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