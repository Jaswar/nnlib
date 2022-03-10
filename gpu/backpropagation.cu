//
// Created by Jan Warchocki on 10/03/2022.
//

#include "backpropagation.cuh"
#include "verify.cuh"
#include "allocation_gpu.cuh"

#ifdef HAS_CUDA

#include <cuda.h>

__global__
void performBackpropagation(DTYPE* biases, DTYPE* weights, DTYPE* data, DTYPE* derivatives,
                            DTYPE* delta, DTYPE* previousWeights, DTYPE* newDelta, int inSize, int outSize, int deltaSize,
                            DTYPE learningRate, bool isLastLayer) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

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

Vector backpropagation(Layer& layer, const Vector& delta, const Matrix& previousWeights,
              bool isLastLayer, DTYPE learningRate) {
    const Vector& derivatives = layer.calculateDerivatives();

    DTYPE* biasesDevice = allocate1DArrayDevice(layer.outSize);
    DTYPE* weightsDevice = allocate1DArrayDevice(layer.inSize * layer.outSize);
    DTYPE* dataDevice = allocate1DArrayDevice(layer.inSize);
    DTYPE* derivativesDevice = allocate1DArrayDevice(layer.outSize);

    DTYPE* deltaDevice = allocate1DArrayDevice(delta.n);
    DTYPE* previousWeightsDevice = allocate1DArrayDevice(previousWeights.n * previousWeights.m);

    DTYPE* newDeltaDevice = allocate1DArrayDevice(layer.outSize);

    copy1DFromHostToDevice(layer.biases.data, biasesDevice, layer.outSize);
    copy2DFromHostToDevice(layer.weights.data, weightsDevice, layer.weights.n, layer.weights.m);
    copy1DFromHostToDevice(layer.data.data, dataDevice, layer.inSize);
    copy1DFromHostToDevice(derivatives.data, derivativesDevice, derivatives.n);

    copy1DFromHostToDevice(delta.data, deltaDevice, delta.n);
    copy2DFromHostToDevice(previousWeights.data, weightsDevice, previousWeights.n, previousWeights.m);

    performBackpropagation<<<1, layer.outSize>>>(biasesDevice, weightsDevice, dataDevice, derivativesDevice,
                                                 deltaDevice, previousWeightsDevice, newDeltaDevice, layer.inSize, layer.outSize, delta.n,
                                                 learningRate, isLastLayer);
    cudaDeviceSynchronize();

    copy1DFromDeviceToHost(biasesDevice, layer.biases.data, layer.outSize);
    copy2DFromDeviceToHost(weightsDevice, layer.weights.data, layer.outSize, layer.inSize);

    DTYPE* newDelta = allocate1DArray(layer.outSize);
    copy1DFromDeviceToHost(newDeltaDevice, newDelta, layer.outSize);

    free1DArrayDevice(biasesDevice);
    free1DArrayDevice(weightsDevice);
    free1DArrayDevice(dataDevice);
    free1DArrayDevice(derivativesDevice);

    free1DArrayDevice(deltaDevice);
    free1DArrayDevice(previousWeightsDevice);

    free1DArrayDevice(newDeltaDevice);

    return Vector(newDelta, layer.outSize);
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
        }
        return newDelta;
    }
}

#endif