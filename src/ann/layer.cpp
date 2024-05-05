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
sTensor initializeBiases(size_t outSize) {
    sTensor biases = std::make_shared<Tensor>(outSize);

    for (int i = 0; i < outSize; i++) {
        biases->data[i] = getRandom();
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
sTensor initializeWeights(size_t inSize, size_t outSize) {
    sTensor weights = std::make_shared<Tensor>(inSize, outSize);

    for (int i = 0; i < inSize; i++) {
        for (int j = 0; j < outSize; j++) {
            weights->data[i * outSize + j] = getRandom();
        }
    }

    return weights;
}

Layer::Layer(size_t inSize, size_t outSize, const std::string& activation, DataLocation location)
    : location(location),
      inSize(inSize),
      outSize(outSize),
      activation(activation),
      biases(initializeBiases(outSize)),
      weights(initializeWeights(inSize, outSize)) {

    if (location == DEVICE) {
        biases->move(DEVICE);
        weights->move(DEVICE);
    }
    weights->useGrad();
    biases->useGrad();
}

Layer::~Layer() = default;

sTensor Layer::forward(const sTensor& batch) {
    sTensor z = add(multiply(batch, weights), biases);

    if (activation == "relu") {
        return relu(z);
    } else if (activation == "sigmoid") {
        return sigmoid(z);
    } else {
        return z;
    }
}

void Layer::applyGradients(size_t batchSize, float learningRate) {
    biases = subtract(biases, multiply(biases->grad, learningRate / static_cast<float>(batchSize)));
    weights = subtract(weights, multiply(weights->grad, learningRate / static_cast<float>(batchSize)));
    biases->useGrad();
    weights->useGrad();
}