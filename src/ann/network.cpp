//
// Created by Jan Warchocki on 03/03/2022.
//

#include "../../include/network.h"
#include "../gpu/allocation_gpu.cuh"
#include <algorithm>
#include <cmath>
#include <ctime>
#include <exceptions/size_mismatch_exception.h>
#include <utils/printing.h>
#include <iomanip>
#include <chrono>

std::vector<Matrix> splitIntoBatches(const Matrix& matrix, size_t batchSize, DataLocation location) {
    std::vector<Matrix> result;

    int numBatches = std::ceil(static_cast<double>(matrix.n) / static_cast<double>(batchSize));
    for (int i = 0; i < numBatches; i++) {
        auto rowsInBatch = std::min(batchSize, matrix.n - batchSize * i);

        DTYPE* allocated;
        if (location == DEVICE) {
            allocated = copy1DArrayDevice(matrix.m * rowsInBatch, &matrix.data[i * matrix.m * batchSize]);
        } else {
            allocated = copy1DArray(matrix.m * rowsInBatch, &matrix.data[i * matrix.m * batchSize]);
        }
        Matrix batch = Matrix(allocated, rowsInBatch, matrix.m, location);

        result.push_back(batch);
    }

    return result;
}

Network::Network(size_t inputSize, bool useGPU, long long seed)
    : seed(seed), layers(), location(HOST), previousSize(inputSize), loss(DEFAULT_BATCH_SIZE, inputSize) {
    if (this->seed == NO_SEED) {
        this->seed = time(nullptr);
    }
    srand(this->seed);

    if (useGPU && isCudaAvailable()) {
        location = DEVICE;
        loss.moveToDevice();
    }
}

void Network::add(size_t numNeurons, const std::string& activation) {
    Activation* activationFunction;
    if (activation == "relu") {
        activationFunction = new ReLUActivation();
    } else if (activation == "sigmoid") {
        activationFunction = new SigmoidActivation();
    } else {
        activationFunction = new LinearActivation();
    }

    Layer newLayer = Layer(previousSize, numNeurons, activationFunction, location);

    layers.push_back(newLayer);

    previousSize = numNeurons;
    loss = Matrix(DEFAULT_BATCH_SIZE, numNeurons, location);
}

Matrix* Network::forward(const Matrix& batch) {
    Layer& first = layers.front();
    first.forward(batch);

    for (auto i = layers.begin() + 1; i != layers.end(); i++) {
        size_t index = i - layers.begin();
        i->forward(layers.at(index - 1).aMatrix);
    }

    return &layers.back().aMatrix;
}

void Network::backward(const Matrix& predicted, const Matrix& target, DTYPE learningRate) {
    if (loss.n != predicted.n) {
        loss = Matrix(predicted.n, loss.m, loss.location);
    }

    // Squared error loss used here
    subtract(predicted, target, loss);

    Layer& last = layers.back();
    last.backward(loss, Matrix(0, 0, location), predicted.n, true);

    for (auto i = layers.rbegin() + 1; i != layers.rend(); ++i) {
        size_t index = i - layers.rbegin();
        Layer& prev = layers.at(layers.size() - index);

        i->backward(prev.newDelta, prev.weights, predicted.n, false);
    }

    for (auto& layer : layers) {
        layer.applyGradients(predicted.n, learningRate);
    }
}

bool moveToHost(Matrix& expected, Matrix& predictions) {
    if (predictions.location == DEVICE) {
        predictions.moveToHost();
        return true;
    }
    return false;
}

int computeCorrect(Matrix& expected, Matrix& predictions, size_t start) {
    bool shouldRestoreToDevice = moveToHost(expected, predictions);

    // Calculate the accuracy on the training set.
    int correct = 0;
    for (int row = 0; row < predictions.n; row++) {
        int maxInx = 0;
        for (int i = 0; i < predictions.m; i++) {
            if (predictions(row, i) > predictions(row, maxInx)) {
                maxInx = i;
            }
        }

        if (expected(start + row, maxInx) == 1) {
            correct++;
        }
    }
    if (shouldRestoreToDevice) {
        predictions.moveToDevice();
    }

    return correct;
}

void displayEpochProgress(size_t processedRows, size_t totalRows, size_t milliseconds, double accuracy) {
    std::cout << "\r"
              << constructProgressBar(processedRows, totalRows) << " "
              << constructPercentage(processedRows, totalRows) << " "
              << constructTime(milliseconds) << ": accuracy = "
              << std::setprecision(3) << accuracy << std::flush;
}

//NOLINTNEXTLINE(readability-identifier-naming)
void Network::train(const Matrix& X, const Matrix& y, int epochs, size_t batchSize, DTYPE learningRate) {
    if (X.n != y.n) {
        throw SizeMismatchException();
    }

    std::vector<Matrix> batches = splitIntoBatches(X, batchSize, location);
    std::vector<Matrix> targets = splitIntoBatches(y, batchSize, location);

    Matrix yHost = y;
    yHost.moveToHost();

    for (int epoch = 1; epoch <= epochs; epoch++) {
        std::cout << "Epoch: " << epoch << std::endl;

        processEpoch(batches, targets, yHost, learningRate);

        std::cout << std::endl;
    }
}

void Network::processEpoch(std::vector<Matrix>& batches, std::vector<Matrix>& targets, Matrix& yHost, DTYPE learningRate) {
    int correct = 0;
    int total = 0;
    auto epochStart = std::chrono::steady_clock::now();

    for (int row = 0; row < batches.size(); row++) {
        const Matrix& batch = batches.at(row);
        Matrix& target = targets.at(row);

        Matrix* output = forward(batch);

        correct += computeCorrect(yHost, *output, row * batch.n);
        total += static_cast<int>(batch.n);

        backward(*output, target, learningRate);

        auto batchEnd = std::chrono::steady_clock::now();
        size_t milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(batchEnd - epochStart).count();

        displayEpochProgress((row + 1) * batch.n, yHost.n, milliseconds, static_cast<double>(correct) / total);
    }

    auto epochEnd = std::chrono::steady_clock::now();
    size_t milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(epochEnd - epochStart).count();

    displayEpochProgress(yHost.n, yHost.n, milliseconds, static_cast<double>(correct) / total);
}

