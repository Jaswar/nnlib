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
 * @brief Generate a random float value.
 *
 * Currently the method only returns uniformly distributed numbers in the range (-0.2, 0.2).
 *
 * @return A random value.
 */
float getRandom() {
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
        biases.data[i] = getRandom();
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
            weights.data[i * outSize + j] = getRandom();
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
      data(0, 0),
      zMatrix(0, 0),
      weightsGradients(inSize, outSize),
      biasesGradients(outSize) {

    if (location == DEVICE) {
        biases.move(DEVICE);
        weights.move(DEVICE);
        zMatrix.move(DEVICE);
        data.move(DEVICE);
        weightsGradients.move(DEVICE);
        biasesGradients.move(DEVICE);
    }
}

Layer::~Layer() = default;

Tensor Layer::forward(const Tensor& batch) {
    zMatrix = multiply(batch, weights);
    zMatrix = add(zMatrix, biases);

    data = batch;

    Tensor aMatrix = activation->forward(zMatrix);
    return aMatrix;
}

Tensor Layer::backward(const Tensor& upstream) {
    Tensor derivatives = this->activation->computeDerivatives(zMatrix);
    Tensor downstream = hadamard(upstream, derivatives);

    weightsGradients = multiply(transpose(data), downstream);
    Tensor ones = Tensor(upstream.shape[0]);
    ones.move(location);
    fill(1.0f, ones);
    biasesGradients = multiply(transpose(downstream), ones);

    downstream = multiply(downstream, transpose(weights));
    return downstream;
}

void Layer::applyGradients(size_t batchSize, float learningRate) {
    biasesGradients = multiply(biasesGradients, learningRate / static_cast<float>(batchSize));
    biases = subtract(biases, biasesGradients);

    weightsGradients = multiply(weightsGradients, learningRate / static_cast<float>(batchSize));
    weights = subtract(weights, weightsGradients);
}