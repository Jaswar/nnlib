/**
 * @file categorical_cross_entropy.cpp
 * @brief Source file defining the methods of the CategoricalCrossEntropy class.
 * @author Jan Warchocki
 * @date 03 January 2023
 */

#include <loss.h>

float CategoricalCrossEntropy::calculateLoss(const Tensor& targets, const Tensor& predictions) {
    Tensor ones = Tensor(targets.shape[1], targets.shape[1]);
    ones.move(targets.location);
    fill(1, ones);

    Tensor accumulatedSumsLoss = multiply(predictions, ones);

    ones = Tensor(targets.shape);
    ones.move(targets.location);
    fill(1, ones);
    accumulatedSumsLoss = divide(ones, accumulatedSumsLoss);

    Tensor workingSpace = hadamard(predictions, accumulatedSumsLoss);
    workingSpace = hadamard(targets, log(workingSpace));

    numSamples += targets.shape[0];
    currentTotalMetric += sum(workingSpace) * -1;

    return currentTotalMetric / static_cast<float>(numSamples);
}

Tensor CategoricalCrossEntropy::calculateDerivatives(const Tensor& targets, const Tensor& predictions) {
    Tensor ones = Tensor(targets.shape[1], targets.shape[1]);
    ones.move(targets.location);
    fill(1, ones);

    Tensor accumulatedSumsDerivatives = multiply(predictions, ones);

    Tensor result = Tensor(targets.shape);
    result.move(targets.location);
    fill(1, result);
    accumulatedSumsDerivatives = divide(result, accumulatedSumsDerivatives);

    result = divide(targets, predictions);
    result = multiply(result, -1);

    result = add(result, accumulatedSumsDerivatives);
    return result;
}

std::string CategoricalCrossEntropy::getShortName() const {
    return "categorical_cross_entropy";
}
