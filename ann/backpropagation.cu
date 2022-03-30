//
// Created by Jan Warchocki on 10/03/2022.
//

#include "backpropagation.cuh"
#include "../gpu/verify.cuh"
#include "../gpu/allocation_gpu.cuh"
#include "../gpu/assert.cuh"

#ifdef HAS_CUDA

__global__
void performBackpropagation(DTYPE* biasesGradients, DTYPE* weightsGradients, const DTYPE* data, const DTYPE* derivatives,
                            const DTYPE* delta, const DTYPE* previousWeights, DTYPE* newDelta, int inSize, int outSize, int deltaSize,
                            DTYPE learningRate, bool isLastLayer, int row) {
    auto index = threadIdx.x;

    if (index >= outSize) {
        return;
    }

    if (isLastLayer) {
        DTYPE coreGradient = delta[row * deltaSize + index] * derivatives[row * outSize + index];

        biasesGradients[index] += learningRate * coreGradient;

        for (int j = 0; j < inSize; j++) {
            weightsGradients[j * outSize + index] += learningRate * coreGradient * data[row * inSize + j];
        }

        newDelta[row * outSize + index] = coreGradient;
    } else {
        DTYPE coreGradient = 0;
        for (int j = 0; j < deltaSize; j++) {
            coreGradient += delta[row * deltaSize + j] * derivatives[row * outSize + index] * previousWeights[index * deltaSize + j];
        }

        biasesGradients[index] += learningRate * coreGradient;

        for (int j = 0; j < inSize; j++) {
            weightsGradients[j * outSize + index] += learningRate * coreGradient * data[row * inSize + j];
        }

        newDelta[row * outSize + index] = coreGradient;
    }
}

void backpropagation(Layer& layer, const Matrix& delta, const Matrix& previousWeights,
              bool isLastLayer, DTYPE learningRate) {
    for (int i = 0; i < 32; i++) {
        cudaDeviceSynchronize();
        performBackpropagation<<<1, layer.outSize>>>(layer.biasesGradients.data, layer.weightsGradients.data,
                                                     layer.data->data, layer.derivatives.data, delta.data,
                                                     previousWeights.data, layer.newDelta.data,
                                                     layer.inSize, layer.outSize, delta.m, learningRate, isLastLayer, i);
    }
}

__global__
void applyGradientsDevice(DTYPE* biases, DTYPE* weights, DTYPE* biasesGradients, DTYPE* weightsGradients, int inSize, int outSize) {
    auto index = threadIdx.x;

    if (index >= outSize) {
        return;
    }

    biases[index] -= biasesGradients[index] / 32;
    biasesGradients[index] = 0;

    for (int i = 0; i < inSize; i++) {
        weights[i * outSize + index] -= weightsGradients[i * outSize + index] / 32;
        weightsGradients[i * outSize + index] = 0;
    }
}

void applyGradient(Layer& layer) {
//    Vector b = layer.biases;
//    b.moveToHost();
//    std::cout << b << std::endl;
//
//    Vector bg = layer.biasesGradients;
//    bg.moveToHost();
//    std::cout << bg << std::endl;
//
//    Matrix w = layer.weights;
//    w.moveToHost();
//    std::cout << w << std::endl;

    applyGradientsDevice<<<1, layer.outSize>>>(layer.biases.data, layer.weights.data,
                                               layer.biasesGradients.data, layer.weightsGradients.data,
                                               layer.inSize, layer.outSize);

//    Vector ub = layer.biases;
//    ub.moveToHost();
//    std::cout << ub << std::endl;
//
//    Matrix uw = layer.weights;
//    uw.moveToHost();
//    std::cout << uw << std::endl;
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