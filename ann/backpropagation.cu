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
    auto index = threadIdx.x;

    if (index >= outSize) {
        return;
    }

    for (int row = 0; row < batchSize; row++) {
        if (isLastLayer) {
            DTYPE coreGradient = delta[row * deltaSize + index] * derivatives[row * outSize + index];

            biasesGradients[index] += coreGradient;

            for (int j = 0; j < inSize; j++) {
                weightsGradients[j * outSize + index] += coreGradient * data[row * inSize + j];
            }

            newDelta[row * outSize + index] = coreGradient;
        } else {
            DTYPE coreGradient = 0;
            for (int j = 0; j < deltaSize; j++) {
                coreGradient += delta[row * deltaSize + j] * derivatives[row * outSize + index] * previousWeights[index * deltaSize + j];
            }

            biasesGradients[index] += coreGradient;

            for (int j = 0; j < inSize; j++) {
                weightsGradients[j * outSize + index] += coreGradient * data[row * inSize + j];
            }

            newDelta[row * outSize + index] = coreGradient;
        }
    }

}

void computeGradients(Layer& layer, const Matrix& delta, const Matrix& previousWeights,
                     int batchSize, bool isLastLayer) {
    computeGradientsDevice<<<1, layer.outSize>>>(layer.biasesGradients.data, layer.weightsGradients.data,
                                                 layer.data->data, layer.derivatives.data, delta.data,
                                                 previousWeights.data, layer.newDelta.data,
                                                 layer.inSize, layer.outSize, delta.m, batchSize, isLastLayer);
    gpuCheckError( cudaDeviceSynchronize() )
}

__global__
void applyGradientsDevice(DTYPE* biases, DTYPE* weights, DTYPE* biasesGradients, DTYPE* weightsGradients,
                          int inSize, int outSize, int batchSize, DTYPE learningRate) {
    auto index = threadIdx.x;

    if (index >= outSize) {
        return;
    }

    biases[index] -= learningRate * biasesGradients[index] / (DTYPE) batchSize;
    biasesGradients[index] = 0;

    for (int i = 0; i < inSize; i++) {
        weights[i * outSize + index] -= learningRate * weightsGradients[i * outSize + index] / (DTYPE) batchSize;
        weightsGradients[i * outSize + index] = 0;
    }
}

void applyGradients(Layer& layer, int batchSize, DTYPE learningRate) {
    applyGradientsDevice<<<1, layer.outSize>>>(layer.biases.data, layer.weights.data,
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