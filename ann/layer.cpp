//
// Created by Jan Warchocki on 03/03/2022.
//

#include <utility>
#include "layer.h"
#include "activation.cuh"
#include "../gpu/backpropagation.cuh"
#include "../gpu/verify.cuh"
#include "../gpu/allocation_gpu.cuh"

DTYPE getRandomValue() {
    // TODO: For large networks, the values at neurons can grow very large rendering them useless
    // A fix can lower the initial weights and biases.
    return (((DTYPE) rand() / RAND_MAX) * 2 - 1) / 5;
}

Vector initializeBiases(int outSize) {
    DTYPE* biases = allocate1DArray(outSize);

    for (int i = 0; i < outSize; i++) {
        biases[i] = getRandomValue();
    }

    return Vector(biases, outSize);
}

Matrix initializeWeights(int inSize, int outSize) {
    Matrix weights = Matrix(outSize, inSize);

    for (int i = 0; i < outSize; i++) {
        for (int j = 0; j < inSize; j++) {
            weights(i, j) = getRandomValue();
        }
    }

    return weights;
}

Layer::Layer(int inSize, int outSize, const std::string& activation)
        : inSize(inSize), outSize(outSize),
          activation(activation),
          biases(initializeBiases(outSize)),
          weights(initializeWeights(inSize, outSize)),
          data(inSize),
          aVector(outSize),
          zVector(outSize),
          newDelta(outSize),
          derivatives(outSize) {
    biases.moveToDevice();
    weights.moveToDevice();

    aVector.moveToDevice();
    zVector.moveToDevice();

    newDelta.moveToDevice();
    derivatives.moveToDevice();
}

Layer::~Layer() = default;

void Layer::forward(const Vector& input) {
    multiply(weights, input, aVector);
    add(aVector, biases, aVector);

    data.data = input.data;

    if (activation == "relu") {
        ReLU(aVector, zVector);
    } else if (activation == "sigmoid") {
        sigmoid(aVector, zVector);
    } else {
        linear(aVector, zVector);
    }
}

void Layer::backward(const Vector& delta, const Matrix& previousWeights,
                                          bool isLastLayer, DTYPE learningRate) {
    calculateDerivatives();

    backpropagation(*this, delta, previousWeights, isLastLayer, learningRate);
}

void Layer::calculateDerivatives() {
    if (activation == "relu") {
        ReLUDerivative(aVector, derivatives);
    } else if (activation == "sigmoid") {
        sigmoidDerivative(aVector, derivatives);
    } else {
        linearDerivative(aVector, derivatives);
    }
}

