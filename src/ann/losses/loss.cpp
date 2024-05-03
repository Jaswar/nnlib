/**
 * @file loss.cpp
 * @brief Source file defining the methods of the Loss class.
 * @author Jan Warchocki
 * @date 25 December 2022
 */

#include <loss.h>

Loss::Loss() : Metric() {
}

float Loss::calculateMetric(const Tensor& targets, const Tensor& predictions) {
    return calculateLoss(targets, predictions);
}

Tensor Loss::calculateDerivatives(const Tensor& targets, const Tensor& predictions) {
    Tensor result = Tensor(predictions.shape);
    result.move(predictions.location);
    calculateDerivatives(targets, predictions, result);
    return result;
}
