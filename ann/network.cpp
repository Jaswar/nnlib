//
// Created by Jan Warchocki on 03/03/2022.
//

#include <time.h>
#include "network.h"
#include "../gpu/allocation_gpu.cuh"
#include <algorithm>

std::vector<Vector> convertToVectors(const Matrix& matrix) {
    std::vector<Vector> result;

    for (int i = 0; i < matrix.n; i++) {
        DTYPE* allocated = copy1DArrayDevice(matrix.m, &matrix.data[i * matrix.m]);
        Vector vector = Vector(allocated, matrix.m, DEVICE);
        result.push_back(vector);
    }

    return result;
}

std::vector<Matrix> splitIntoBatches(const Matrix& matrix, int batchSize, bool doTranspose = false) {
    std::vector<Matrix> result;

    int numBatches = std::ceil(matrix.n / (double) batchSize);
    for (int i = 0; i < numBatches; i++) {
        int rowsInBatch = std::min(batchSize, static_cast<int>(matrix.n - batchSize * i));
        DTYPE* allocated = copy1DArrayDevice(matrix.m * rowsInBatch, &matrix.data[i * matrix.m * batchSize]);
        Matrix batch = Matrix(allocated, rowsInBatch, matrix.m, DEVICE);

        if (doTranspose) {
            DTYPE* allocatedT = allocate1DArrayDevice(matrix.m * rowsInBatch);
            Matrix batchT = Matrix(allocatedT, matrix.m, rowsInBatch, DEVICE);
            transpose(batch, batchT);

            result.push_back(batchT);
        } else {
            result.push_back(batch);
        }
    }

    return result;
}

Network::Network(int inputSize, long long seed) : seed(seed), layers(), previousSize(inputSize), loss(DEFAULT_BATCH_SIZE, inputSize, DEVICE) {
    if (this->seed == NO_SEED) {
        this->seed = time(nullptr);
    }
    srand(this->seed);
}

void Network::add(int numNeurons, const std::string& activation) {
    Layer newLayer = Layer(previousSize, numNeurons, activation);

    layers.push_back(newLayer);

    previousSize = numNeurons;
    loss = Matrix(DEFAULT_BATCH_SIZE, numNeurons, DEVICE);
}

Matrix* Network::forward(const Matrix& input) {
    Layer& first = layers.front();
    first.forward(input);

    for (auto i = layers.begin() + 1; i != layers.end(); i++) {
        size_t index = i - layers.begin();
        i->forward(layers.at(index - 1).zMatrix);
    }

    return &layers.back().zMatrix;
}

void Network::backward(const Matrix& predicted, const Matrix& target, DTYPE learningRate) {
    if (loss.n != predicted.n) {
        loss = Matrix(predicted.n, loss.m, loss.location);
    }

    // Squared error loss used here
    subtract(predicted, target, loss);

    Layer& last = layers.back();
    last.backward(loss, Matrix(0, 0, DEVICE), predicted.n, true);

    for (auto i = layers.rbegin() + 1; i != layers.rend(); ++i) {
        size_t index = i - layers.rbegin();
        Layer& prev = layers.at(layers.size() - index);

        i->backward(prev.newDelta, prev.weights, predicted.n, false);
    }

    for (auto& layer : layers) {
        layer.applyGradients(predicted.n, learningRate);
    }
}

void Network::train(const Matrix& X, const Matrix& y, int epochs, int batchSize, DTYPE learningRate) {
    if (X.n != y.n) {
        throw SizeMismatchException();
    }

    Matrix yCopy = y;
    yCopy.moveToHost();

    std::vector<Matrix> batches = splitIntoBatches(X, batchSize);
    std::vector<Matrix> targets = splitIntoBatches(y, batchSize);

    for (int epoch = 1; epoch <= epochs; epoch++) {
        std::cout << "Epoch: " << epoch << std::endl;
        for (int row = 0; row < batches.size(); row++) {
            const Matrix* output = forward(batches.at(row));

            backward(*output, targets.at(row), learningRate);
        }

        // Calculate the accuracy on the training set.
        int correct = 0;
        for (int batch = 0; batch < batches.size(); batch++) {
            const Matrix* output = forward(batches.at(batch));
            Matrix copy = *output;
            copy.moveToHost();

            for (int row = 0; row < batchSize; row++) {
                if (batch * batchSize + row >= X.n) {
                    continue;
                }

                int maxInx = 0;
                for (int i = 0; i < y.m; i++) {
                    if (copy(row, i) > copy(row, maxInx)) {
                        maxInx = i;
                    }
                }

                if (yCopy(batch * batchSize + row, maxInx) == 1) {
                    correct++;
                }
            }
        }
        std::cout << ((double) correct) / X.n << std::endl;
    }
}
