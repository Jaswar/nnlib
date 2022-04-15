//
// Created by Jan Warchocki on 03/03/2022.
//

#include <utility>
#include "layer.h"
#include "activation.cuh"
#include "backpropagation.cuh"
#include "../gpu/verify.cuh"
#include "../gpu/allocation_gpu.cuh"

DTYPE getRandomValue() {
    // TODO: For large networks, the values at neurons can grow very large rendering them useless
    // A fix can lower the initial weights and biases.
    return (((DTYPE) rand() / RAND_MAX) * 2 - 1) / 5;
}

Vector initializeBiases(size_t outSize) {
    DTYPE* biases = allocate1DArray(outSize);

    for (int i = 0; i < outSize; i++) {
        biases[i] = getRandomValue();
    }

    return Vector(biases, outSize);
}

Matrix initializeWeights(size_t inSize, size_t outSize) {
    Matrix weights = Matrix(inSize, outSize);

    for (int i = 0; i < inSize; i++) {
        for (int j = 0; j < outSize; j++) {
            weights(i, j) = getRandomValue();
        }
    }

    return weights;
}

Layer::Layer(size_t inSize, size_t outSize, std::string activation)
        : inSize(inSize), outSize(outSize),
          activation(std::move(activation)),
          biases(initializeBiases(outSize)),
          weights(initializeWeights(inSize, outSize)),
          data(),
          aMatrix(DEFAULT_BATCH_SIZE, outSize),
          zMatrix(DEFAULT_BATCH_SIZE, outSize),
          newDelta(DEFAULT_BATCH_SIZE, outSize),
          derivatives(DEFAULT_BATCH_SIZE, outSize),
          weightsGradients(allocate1DArray(inSize * outSize, 0), inSize, outSize),
          biasesGradients(allocate1DArray(outSize, 0), outSize) {
    biases.moveToDevice();
    weights.moveToDevice();

    aMatrix.moveToDevice();
    zMatrix.moveToDevice();

    newDelta.moveToDevice();
    derivatives.moveToDevice();

    weightsGradients.moveToDevice();
    biasesGradients.moveToDevice();
}

Layer::~Layer() = default;

void Layer::forward(const Matrix& batch) {
    allocate(batch.n);

    multiply(batch, weights, zMatrix);
    add(zMatrix, biases, zMatrix);

    data = &batch;

    if (activation == "relu") {
        ReLU(zMatrix, aMatrix);
    } else if (activation == "sigmoid") {
        sigmoid(zMatrix, aMatrix);
    } else {
        linear(zMatrix, aMatrix);
    }
}

void Layer::backward(const Matrix& delta, const Matrix& previousWeights, size_t batchSize, bool isLastLayer) {
    calculateDerivatives();

    computeGradients(*this, delta, previousWeights, batchSize, isLastLayer);
}

void Layer::applyGradients(size_t batchSize, DTYPE learningRate) {
    ::applyGradients(*this, batchSize, learningRate);
}

void Layer::calculateDerivatives() {
    if (activation == "relu") {
        ReLUDerivative(zMatrix, derivatives);
    } else if (activation == "sigmoid") {
        sigmoidDerivative(zMatrix, derivatives);
    } else {
        linearDerivative(zMatrix, derivatives);
    }
}

void Layer::allocate(size_t batchSize) {
    if (aMatrix.n != batchSize) {
        aMatrix = Matrix(batchSize, aMatrix.m, aMatrix.location);
    }
    if (zMatrix.n != batchSize) {
        zMatrix = Matrix(batchSize, zMatrix.m, zMatrix.location);
    }
    if (newDelta.n != batchSize) {
        newDelta = Matrix(batchSize, newDelta.m, newDelta.location);
    }
    if (derivatives.n != batchSize) {
        derivatives = Matrix(batchSize, derivatives.m, derivatives.location);
    }
}
