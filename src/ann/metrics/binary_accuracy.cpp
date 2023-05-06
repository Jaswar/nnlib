/**
 * @file binary_accuracy.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 07 May 2023
 */

#include "exceptions/unsupported_operation_exception.h"
#include <metric.h>

BinaryAccuracy::BinaryAccuracy() : Metric() {

}

bool isCorrectPrediction(const float expected, const float actual) {
    return (expected == 1 && actual >= 0.5) || (expected == 0 && actual < 0.5);
}

float BinaryAccuracy::calculateMetric(const Tensor& targets, const Tensor& predictions) {
    if (targets.shape.size() != 2 || targets.shape[1] != 1 || targets.shape != predictions.shape) {
        throw UnsupportedOperationException();
    }

    int correctSamples = 0;
    for (size_t sample = 0; sample < targets.shape[0]; sample++) {
        float target = targets.data[sample];
        float prediction = predictions.data[sample];

        if (isCorrectPrediction(target, prediction)) {
            correctSamples += 1;
        }
    }

    currentTotalMetric += static_cast<float>(correctSamples);
    numSamples += targets.shape[0];

    return currentTotalMetric / static_cast<float>(numSamples);
}
std::string BinaryAccuracy::getShortName() const {
    return "binary_accuracy";
}
