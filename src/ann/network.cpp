/**
 * @file network.cpp
 * @brief Source file defining methods of the Network class.
 * @author Jan Warchocki
 * @date 03 March 2022
 */

#include "../../include/network.h"
#include "../gpu/allocation_gpu.cuh"
#include "runtime.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <exceptions/size_mismatch_exception.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <utils/printing.h>
#include <verify.cuh>

/**
 * @brief Structure to contain information about the current epoch.
 */
struct EpochProgress {

    /**
     * @brief Total number of samples already processed.
     */
    size_t numProcessed;

    /**
     * @brief The total number of samples (processed + non-processed).
     */
    size_t numTotal;

    /**
     * @brief The value of the loss.
     */
    float lossValue;

    /**
     * @brief Mapping of names of metrics to their computed values.
     */
    std::map<std::string, float> metricsValues;

    /**
     * @brief The time the epoch started.
     */
    std::chrono::time_point<std::chrono::steady_clock> timeStart;

    /**
     * @brief The constructor of the EpochProgress class.
     *
     * @param numTotal The total number of samples to be processed.
     */
    explicit EpochProgress(size_t numTotal)
        : numProcessed(0), numTotal(numTotal), lossValue(0), metricsValues(), timeStart() {
        timeStart = std::chrono::steady_clock::now();
    }

    /**
     * @brief Update the current state of the epoch.
     *
     * @param targets The desired predictions of the network in the current batch.
     * @param predictions The actual predictions of the network in the current batch.
     * @param loss The loss object to update.
     * @param metrics The metrics to update given the new batch.
     */
    void update(sfTensor& targets, sfTensor& predictions, Loss* loss, std::vector<Metric*>& metrics) {
        const DataLocation originalPredictions = predictions->location;
        const DataLocation originalTargets = targets->location;

        predictions->move(HOST);
        targets->move(HOST);

        numProcessed += targets->shape[0];

        Runtime::getInstance().disableGradient();
        lossValue = loss->calculateMetric(targets, predictions);
        for (auto metric : metrics) {
            float metricValue = metric->calculateMetric(targets, predictions);
            metricsValues[metric->getShortName()] = metricValue;
        }
        Runtime::getInstance().enableGradient();

        predictions->move(originalPredictions);
        targets->move(originalTargets);
    }
};

/**
 * @brief A toString() method for EpochProgress.
 *
 * @param stream The stream to append the string representation of EpochProgress to.
 * @param epochProgress The EpochProgress object to display.
 * @return The string representation of the desired EpochProgress object.
 */
std::ostream& operator<<(std::ostream& stream, const EpochProgress& epochProgress) {
    auto timeNow = std::chrono::steady_clock::now();
    size_t milliseconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(timeNow - epochProgress.timeStart).count();

    stream << "\r" << constructProgressBar(epochProgress.numProcessed, epochProgress.numTotal) << " "
           << constructPercentage(epochProgress.numProcessed, epochProgress.numTotal) << " "
           << constructTime(milliseconds) << ": loss = " << std::setprecision(3) << epochProgress.lossValue;
    for (auto const& entry : epochProgress.metricsValues) {
        stream << "; " << entry.first << " = " << entry.second;
    }
    stream << std::flush;
    return stream;
}

/**
 * @brief Split the input matrix into batches.
 *
 * It is assumed that the data samples are row-aligned. Therefore, to split into batches, the input matrix is
 * divided along the row axis, so that the data samples are not split.
 *
 * For example, when splitting the following matrix using @p batchSize = 2:
 * ```
 * [[1, 1],
 *  [2, 2],
 *  [3, 3],
 *  [4, 4]]
 * ```
 * We get as the result a vector of the following two matrices:
 * ```
 * [[1, 1],
 *  [2, 2]]
 * ```
 * ```
 * [[3, 3],
 *  [4, 4]]
 * ```
 *
 * If the size of @p data is not divisible by @p batchSize, the last batch will be smaller than the rest.
 *
 * @param data The matrix to split into batches.
 * @param batchSize The size of the batch to split on.
 * @return The vector of batches.
 */
std::vector<sfTensor> splitIntoBatches(const sfTensor& data, size_t batchSize) {
    std::vector<sfTensor> result;

    int numBatches = std::ceil(static_cast<double>(data->shape[0]) / static_cast<double>(batchSize));
    for (int i = 0; i < numBatches; i++) {
        auto rowsInBatch = std::min(batchSize, data->shape[0] - batchSize * i);

        sfTensor batch = std::make_shared<Tensor<float>>(rowsInBatch, data->shape[1]);
        if (data->location == DEVICE) {
            copy1DFromDeviceToHost(data->data + i * data->shape[1] * batchSize, batch->data,
                                   data->shape[1] * rowsInBatch);
        } else {
            copy1DFromHostToHost(data->data + i * data->shape[1] * batchSize, batch->data,
                                 data->shape[1] * rowsInBatch);
        }
        batch->move(data->location);

        result.push_back(batch);
    }

    return result;
}

Network::Network(size_t inputSize, bool useGPU, long long seed)
    : seed(seed), layers(), location(HOST), previousSize(inputSize) {
    if (this->seed == NO_SEED) {
        this->seed = time(nullptr);
    }
    srand(this->seed);

    if (useGPU && isCudaAvailable()) {
        location = DEVICE;
    }
}

void Network::add(size_t numNeurons, const std::string& activation) {
    Layer newLayer = Layer(previousSize, numNeurons, activation, location);

    layers.push_back(newLayer);

    previousSize = numNeurons;
}

sfTensor Network::forward(const sfTensor& batch) {
    Layer& first = layers.front();
    sfTensor output = first.forward(batch);

    for (auto i = layers.begin() + 1; i != layers.end(); i++) {
        output = i->forward(output);
    }

    return output;
}

//NOLINTNEXTLINE(readability-identifier-naming)
void Network::train(sfTensor& X, sfTensor& y, int epochs, size_t batchSize, float learningRate, Loss* loss,
                    std::vector<Metric*>& metrics) {
    if (X->shape[0] != y->shape[0]) {
        throw SizeMismatchException();
    }

    X->move(location);
    y->move(location);

    std::vector<sfTensor> batches = splitIntoBatches(X, batchSize);
    std::vector<sfTensor> targets = splitIntoBatches(y, batchSize);
    std::vector<sfTensor> targetsOnHost = std::vector<sfTensor>();
    for (sfTensor& batch : targets) {
        sfTensor batchOnHost = batch->copy();
        batchOnHost->move(HOST);
        targetsOnHost.push_back(batchOnHost);
    }

    for (int epoch = 1; epoch <= epochs; epoch++) {
        std::cout << "Epoch: " << epoch << "/" << epochs << std::endl;

        loss->reset();
        for (auto metric : metrics) {
            metric->reset();
        }
        processEpoch(batches, targets, targetsOnHost, learningRate, loss, metrics);

        std::cout << std::endl;
    }
}

void Network::processEpoch(std::vector<sfTensor>& batches, std::vector<sfTensor>& targets,
                           std::vector<sfTensor>& targetsOnHost, float learningRate, Loss* loss,
                           std::vector<Metric*>& metrics) {
    size_t numSamples = 0;
    for (sfTensor& batch : targets) {
        numSamples += batch->shape[0];
    }
    EpochProgress epochProgress = EpochProgress(numSamples);

    for (int row = 0; row < batches.size(); row++) {
        sfTensor batch = batches.at(row);
        sfTensor target = targets.at(row);

        sfTensor output = forward(batch);

        sfTensor l = loss->calculateLoss(target, output);
        l->backward();

        for (auto& layer : layers) {
            layer.applyGradients(batch->shape[0], learningRate);
        }

        sfTensor targetOnHost = targetsOnHost.at(row);
        epochProgress.update(targetOnHost, output, loss, metrics);
        std::cout << epochProgress;
    }
}
