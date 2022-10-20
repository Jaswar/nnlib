/**
 * @file layer.cpp
 * @brief Source file defining methods of the Layer class.
 * @author Jan Warchocki
 * @date 03 March 2022
 */

#include "../../include/layer.h"
#include "../gpu/allocation_gpu.cuh"
#include "verify.cuh"
#include <utility>

/**
 * @brief Generate a random #DTYPE value.
 *
 * Currently the method only returns uniformly distributed numbers in the range (-0.2, 0.2).
 *
 * @return A random value.
 */
float getRandomValue() {
    // TODO: For large networks, the values at neurons can grow very large rendering them useless
    // A fix can lower the initial weights and biases.
    // For now don't lint it and use rand()
    // NOLINTNEXTLINE
    return (((float) rand() / RAND_MAX) * 2 - 1) / 5;
}

/**
 * @brief Initialize the biases of the layer.
 *
 * This creates a vector of random numbers using the getRandomValue() method.
 *
 * @param outSize The size of the vector to generate. It is also the output size of the layer.
 * @return The random vector of biases.
 */
Tensor initializeBiases(size_t outSize) {
    Tensor biases = Tensor(outSize);

    for (int i = 0; i < outSize; i++) {
        biases.host[i] = getRandomValue();
    }

    return biases;
}

/**
 * @brief Initialize the weights of the layer.
 *
 * This creates a matrix of random numbers using the getRandomValue() method.
 *
 * @param inSize The number of rows of the matrix. It is also the input size to the layer.
 * @param outSize The number of columns of the matrix. It is also the output size of the layer.
 * @return The random matrix of weights.
 */
Tensor initializeWeights(size_t inSize, size_t outSize) {
    Tensor weights = Tensor(inSize, outSize);

    for (int i = 0; i < inSize; i++) {
        for (int j = 0; j < outSize; j++) {
            weights.host[i * outSize + j] = getRandomValue();
        }
    }

    return weights;
}

Layer::Layer(size_t inSize, size_t outSize, Activation* activation, DataLocation location)
    : location(location),
      inSize(inSize),
      outSize(outSize),
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
      weightsGradients(inSize, outSize),
      biasesGradients(outSize),
      ones(DEFAULT_BATCH_SIZE) {

    fill(1, ones);

    if (location == DEVICE) {
        dataT.move(DEVICE);

        biases.move(DEVICE);
        weights.move(DEVICE);

        aMatrix.move(DEVICE);
        zMatrix.move(DEVICE);

        newDelta.move(DEVICE);
        newDeltaT.move(DEVICE);
        derivatives.move(DEVICE);

        previousWeightsT.move(DEVICE);

        weightsGradients.move(DEVICE);
        biasesGradients.move(DEVICE);
        ones.move(DEVICE);
    }
}

Layer::~Layer() = default;

void Layer::forward(const Tensor& batch) {
    allocate(batch.shape[0]);

    multiply(batch, weights, zMatrix);
    add(zMatrix, biases, zMatrix);

    data = &batch;
    transpose(batch, dataT);

    activation->forward(zMatrix, aMatrix);
}

void Layer::backward(const Tensor& delta, const Tensor& previousWeights, size_t batchSize, bool isLastLayer) {
    if (previousWeightsT.shape[0] != previousWeights.shape[1] ||
        previousWeightsT.shape[1] != previousWeights.shape[0]) {
        previousWeightsT = Tensor(previousWeights.shape[1], previousWeights.shape[0]);
        previousWeightsT.move(location);
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
    multiply(biasesGradients, learningRate / static_cast<DTYPE>(batchSize), biasesGradients);
    subtract(biases, biasesGradients, biases);

    multiply(weightsGradients, learningRate / static_cast<DTYPE>(batchSize), weightsGradients);
    subtract(weights, weightsGradients, weights);
}

void Layer::calculateDerivatives() {
    activation->computeDerivatives(zMatrix, derivatives);
}

void Layer::allocate(size_t batchSize) {
    allocateOnes(batchSize);
    allocateDataT(batchSize);
    allocateAMatrix(batchSize);
    allocateZMatrix(batchSize);
    allocateNewDelta(batchSize);
    allocateNewDeltaT(batchSize);
    allocateDerivatives(batchSize);
}

void Layer::allocateOnes(size_t batchSize) {
    if (ones.shape[0] != batchSize) {
        ones = Tensor(batchSize);
        fill(1, ones);
        ones.move(location);
    }
}

void Layer::allocateDataT(size_t batchSize) {
    if (dataT.shape[1] != batchSize) {
        dataT = Tensor(dataT.shape[0], batchSize);
        dataT.move(location);
    }
}

void Layer::allocateAMatrix(size_t batchSize) {
    if (aMatrix.shape[0] != batchSize) {
        aMatrix = Tensor(batchSize, aMatrix.shape[1]);
        aMatrix.move(location);
    }
}

void Layer::allocateZMatrix(size_t batchSize) {
    if (zMatrix.shape[0] != batchSize) {
        zMatrix = Tensor(batchSize, zMatrix.shape[1]);
        zMatrix.move(location);
    }
}

void Layer::allocateNewDelta(size_t batchSize) {
    if (newDelta.shape[0] != batchSize) {
        newDelta = Tensor(batchSize, newDelta.shape[1]);
        newDelta.move(location);
    }
}

void Layer::allocateNewDeltaT(size_t batchSize) {
    if (newDeltaT.shape[1] != batchSize) {
        newDeltaT = Tensor(newDeltaT.shape[0], batchSize);
        newDeltaT.move(location);
    }
}

void Layer::allocateDerivatives(size_t batchSize) {
    if (derivatives.shape[0] != batchSize) {
        derivatives = Tensor(batchSize, derivatives.shape[1]);
        derivatives.move(location);
    }
}
