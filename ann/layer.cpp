//
// Created by Jan Warchocki on 03/03/2022.
//

#include <utility>
#include "layer.h"
#include "activation.h"

DTYPE getRandomValue() {
    return ((DTYPE) rand() / RAND_MAX) * 2 - 1;
}

Vector initializeBiases(int outSize) {
    DTYPE* biases = allocate1DArray(outSize);

    for (int i = 0; i < outSize; i++) {
        biases[i] = getRandomValue();
    }

    return Vector(biases, outSize);
}

Matrix initializeWeights(int inSize, int outSize) {
    DTYPE** weights = allocate2DArray(outSize, inSize);

    for (int i = 0; i < outSize; i++) {
        for (int j = 0; j < inSize; j++) {
            weights[i][j] = getRandomValue();
        }
    }

    return Matrix(weights, outSize, inSize);
}

Layer::Layer(int inSize, int outSize, const std::string& activation)
        : inSize(inSize), outSize(outSize),
          activation(activation),
          biases(initializeBiases(outSize)),
          weights(initializeWeights(inSize, outSize)),
          data(inSize),
          aVector(outSize) {}

Vector Layer::forward(const Vector& input) {
    aVector = weights * input + biases;
    data = input;

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
    const Vector& derivatives = calculateDerivatives();
    if (!isLastLayer) {
        Vector newDelta = Vector(outSize);
        for (int i = 0; i < outSize; i++) {
            DTYPE coreGradient = 0;
            for (int j = 0; j < delta.n; j++) {
                coreGradient += delta[j] * derivatives[i] * previousWeights[j][i];
            }

            biases[i] -= learningRate * coreGradient;

            for (int j = 0; j < inSize; j++) {
                weights[i][j] -= learningRate * coreGradient * data[j];
            }

            newDelta[i] = coreGradient;
        }

        return {newDelta, weights};
    } else {
        Vector newDelta = delta;
        for (int i = 0; i < outSize; i++) {
            DTYPE coreGradient = delta[i] * derivatives[i];

            biases[i] -= learningRate * coreGradient;

            for (int j = 0; j < inSize; j++) {
                weights[i][j] -= learningRate * coreGradient * data[j];
            }
        }
        return {newDelta, weights};
    }
}

Vector Layer::calculateDerivatives() {
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


