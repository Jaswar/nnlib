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

void CategoricalCrossEntropy::calculateDerivatives(const Tensor& targets, const Tensor& predictions,
                                                   Tensor& destination) {
    Tensor ones = Tensor(targets.shape[1], targets.shape[1]);
    ones.move(targets.location);
    fill(1, ones);

    Tensor accumulatedSumsDerivatives = multiply(predictions, ones);

    fill(1, destination);
    divide(destination, accumulatedSumsDerivatives, accumulatedSumsDerivatives);

    divide(targets, predictions, destination);
    multiply(destination, -1, destination);

    add(destination, accumulatedSumsDerivatives, destination);
}

std::string CategoricalCrossEntropy::getShortName() const {
    return "categorical_cross_entropy";
}
