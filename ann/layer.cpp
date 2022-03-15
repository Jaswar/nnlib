//
// Created by Jan Warchocki on 03/03/2022.
//

#include <utility>
#include "layer.h"
#include "activation.h"
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
          aVector(outSize) {}

Layer::~Layer() = default;

Vector Layer::forward(const Vector& input) {
    data.moveToHost();
    weights.moveToHost();
    biases.moveToHost();
    aVector = weights * input + biases;
    data = input;

    data.moveToDevice();

    if (activation == "relu") {
        return ReLU(aVector);
    } else if (activation == "sigmoid") {
        return sigmoid(aVector);
    } else if (activation == "tanh") {
        return tanh(aVector);
    } else {
        return aVector;
    }
}

std::pair<Vector, Matrix> Layer::backward(const Vector& delta, const Matrix& previousWeights,
                                          bool isLastLayer, DTYPE learningRate) {
    biases.moveToDevice();
    weights.moveToDevice();

    const Vector& newDelta = backpropagation(*this, delta, previousWeights, isLastLayer, learningRate);
    return {newDelta, weights};
}

Vector Layer::calculateDerivatives() const {
    if (activation == "relu") {
        return ReLUDerivative(aVector);
    } else if (activation == "sigmoid") {
        return sigmoidDerivative(aVector);
    } else if (activation == "tanh") {
        return tanhDerivative(aVector);
    } else {
        DTYPE* newData = allocate1DArray(aVector.n, 1);
        return Vector(newData, aVector.n);
    }
}


