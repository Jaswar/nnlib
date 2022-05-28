//
// Created by Jan Warchocki on 03/03/2022.
//

#include <utility>
#include "../../include/layer.h"
#include "backpropagation.cuh"
#include "verify.cuh"
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

Layer::Layer(size_t inSize, size_t outSize, Activation* activation)
        : inSize(inSize), outSize(outSize),
          activation(activation),
          biases(initializeBiases(outSize)),
          weights(initializeWeights(inSize, outSize)),
          data(),
          dataT(inSize, DEFAULT_BATCH_SIZE),
          aMatrix(DEFAULT_BATCH_SIZE, outSize),
          zMatrix(DEFAULT_BATCH_SIZE, outSize),
          newDelta(DEFAULT_BATCH_SIZE, outSize),
          newDeltaT(outSize, DEFAULT_BATCH_SIZE),
          derivatives(DEFAULT_BATCH_SIZE, outSize),
          previousWeightsT(0, 0),
          weightsGradients(allocate1DArray(inSize * outSize, 0), inSize, outSize),
          biasesGradients(allocate1DArray(outSize, 0), outSize),
          ones(allocate1DArray(DEFAULT_BATCH_SIZE, 1), DEFAULT_BATCH_SIZE) {

    dataT.moveToDevice();

    biases.moveToDevice();
    weights.moveToDevice();

    aMatrix.moveToDevice();
    zMatrix.moveToDevice();

    newDelta.moveToDevice();
    newDeltaT.moveToDevice();
    derivatives.moveToDevice();

    previousWeightsT.moveToDevice();

    weightsGradients.moveToDevice();
    biasesGradients.moveToDevice();
    ones.moveToDevice();
}

Layer::~Layer() = default;

void Layer::forward(const Matrix& batch) {
    allocate(batch.n);

    multiply(batch, weights, zMatrix);
    add(zMatrix, biases, zMatrix);

    data = &batch;
    transpose(batch, dataT);

    activation->forward(zMatrix, aMatrix);
}

void Layer::backward(const Matrix& delta, const Matrix& previousWeights, size_t batchSize, bool isLastLayer) {
    if (previousWeightsT.n != previousWeights.n || previousWeightsT.m != previousWeights.m) {
        previousWeightsT = Matrix(previousWeights.m, previousWeights.n, previousWeights.location);
    }
    calculateDerivatives();

    if (!isLastLayer) {
        transpose(previousWeights, previousWeightsT);
        multiply(delta, previousWeightsT, newDelta);
        hadamard(newDelta, derivatives, newDelta);

        transpose(newDelta, newDeltaT);
        multiply(newDeltaT, ones, biasesGradients);

        multiply(dataT, newDelta, weightsGradients);
    } else {
        hadamard(delta, derivatives, newDelta);

        transpose(newDelta, newDeltaT);
        multiply(newDeltaT, ones, biasesGradients);

        multiply(dataT, newDelta, weightsGradients);
    }
}

void Layer::applyGradients(size_t batchSize, DTYPE learningRate) {
    ::applyGradients(*this, batchSize, learningRate);
}

void Layer::calculateDerivatives() {
    activation->computeDerivatives(zMatrix, derivatives);
}

void Layer::allocate(size_t batchSize) {
    if (ones.n != batchSize) {
        Vector temp = Vector(allocate1DArray(batchSize, 1), batchSize);
        temp.moveToDevice();
        ones = temp;
    }
    if (dataT.m != batchSize) {
        dataT = Matrix(dataT.n, batchSize, dataT.location);
    }
    if (aMatrix.n != batchSize) {
        aMatrix = Matrix(batchSize, aMatrix.m, aMatrix.location);
    }
    if (zMatrix.n != batchSize) {
        zMatrix = Matrix(batchSize, zMatrix.m, zMatrix.location);
    }
    if (newDelta.n != batchSize) {
        newDelta = Matrix(batchSize, newDelta.m, newDelta.location);
    }
    if (newDeltaT.m != batchSize) {
        newDeltaT = Matrix(newDeltaT.n, batchSize, newDeltaT.location);
    }
    if (derivatives.n != batchSize) {
        derivatives = Matrix(batchSize, derivatives.m, derivatives.location);
    }
}