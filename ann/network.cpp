//
// Created by Jan Warchocki on 03/03/2022.
//

#include <time.h>
#include "network.h"
#include "../gpu/allocation_gpu.cuh"

Network::Network(int inputSize, long long seed) : seed(seed), layers(), previousSize(inputSize) {
    if (this->seed == NO_SEED) {
        this->seed = time(nullptr);
    }
    srand(this->seed);
}

void Network::add(int numNeurons, const std::string& activation) {
    Layer newLayer = Layer(previousSize, numNeurons, activation);

    layers.push_back(newLayer);

    previousSize = numNeurons;
}

Vector Network::forward(const Vector& input) {
    Layer& first = layers.front();
    first.forward(input);

    for (auto i = layers.begin() + 1; i != layers.end(); i++) {
        size_t index = i - layers.begin();
        i->forward(layers.at(index - 1).zVector);
    }

    return layers.back().zVector;
}

void Network::backward(const Vector& predicted, const Vector& target, DTYPE learningRate) {
    // Mean squared error loss used here
    Vector loss = Vector(target.n);
    loss.moveToDevice();

    subtract(predicted, target, loss);
    multiply((1 / (DTYPE) target.n), loss, loss);

    Layer& last = layers.back();
    last.backward(loss, Matrix(0, 0, DEVICE), true, learningRate);

    for (auto i = layers.rbegin() + 1; i != layers.rend(); ++i) {
        size_t index = i - layers.rbegin();
        Layer& prev = layers.at(layers.size() - index);

        i->backward(prev.newDelta, prev.weights, false, learningRate);
    }
}

void Network::train(const Matrix& X, const Matrix& y, int epochs, DTYPE learningRate) {
    if (X.n != y.n) {
        throw SizeMismatchException();
    }

    for (int epoch = 1; epoch <= epochs; epoch++) {
        std::cout << "Epoch: " << epoch << std::endl;
        for (int row = 0; row < X.n; row++) {
            const Vector& input = Vector(copy1DArrayDevice(X.m, &X.data[row * X.m]), X.m, DEVICE);
            const Vector& output = forward(input);

            const Vector& targets = Vector(copy1DArrayDevice(y.m, &y.data[row * y.m]), y.m, DEVICE);
            backward(output, targets, learningRate);
        }

        Matrix yCopy = y;
        yCopy.moveToHost();
        // Calculate the accuracy on the training set.
        int correct = 0;
        for (int row = 0; row < X.n; row++) {
            const Vector& input = Vector(copy1DArrayDevice(X.m, &X.data[row * X.m]), X.m, DEVICE);
            Vector output = forward(input);
            output.moveToHost();

            int maxInx = 0;
            for (int i = 0; i < y.m; i++) {
                if (output[i] > output[maxInx]) {
                    maxInx = i;
                }
            }

            if (yCopy(row, maxInx) == 1) {
                correct++;
            }
        }
        std::cout << ((double) correct) / X.n << std::endl;
    }
}
