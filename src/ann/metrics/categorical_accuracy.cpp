/**
 * @file categorical_accuracy.cpp
 * @brief Source file defining the methods of the CategoricalAccuracy class.
 * @author Jan Warchocki
 * @date 13 February 2023
 */

#include "exceptions/unsupported_operation_exception.h"
#include <loss.h>
#include <metric.h>


CategoricalAccuracy::CategoricalAccuracy() : Metric() {

}

float CategoricalAccuracy::calculateMetric(const Tensor& targets, const Tensor& predictions) {
    if (targets.shape.size() != 2 || predictions.shape.size() != 2 || targets.shape != predictions.shape) {
        throw UnsupportedOperationException();
    }
    size_t batchSize = targets.shape[0];
    size_t numClasses = targets.shape[1];

    int correctSamples = 0;
    for (size_t sample = 0; sample < batchSize; sample++) {
        float maxValue = -std::numeric_limits<float>::max();
        size_t maxIndex = 0;
        for (size_t cls = 0; cls < numClasses; cls++) {
            float prediction = predictions.data[sample * numClasses + cls];
            if (prediction > maxValue) {
                maxValue = prediction;
                maxIndex = cls;
            }
        }

        if (targets.data[sample * numClasses + maxIndex] == 1) {
            correctSamples++;
        }
    }

    currentTotalMetric += static_cast<float>(correctSamples);
    numSamples += batchSize;

    return currentTotalMetric / static_cast<float>(numSamples);
}

std::string CategoricalAccuracy::getShortName() const {
    return "categorical_accuracy";
}
