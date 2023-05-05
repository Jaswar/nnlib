/**
 * @file categorical_cross_entropy.cpp
 * @brief Source file defining the methods of the CategoricalCrossEntropy class.
 * @author Jan Warchocki
 * @date 03 January 2023
 */

#include <loss.h>

float CategoricalCrossEntropy::calculateLoss(const Tensor& targets, const Tensor& predictions) {
    allocateWorkingSpacesLoss(targets, predictions);

    multiply(predictions, onesLoss, accumulatedSumsLoss);
    fill(1, workingSpace);
    divide(workingSpace, accumulatedSumsLoss, accumulatedSumsLoss);

    hadamard(predictions, accumulatedSumsLoss, workingSpace);
    log(workingSpace, workingSpace);
    hadamard(targets, workingSpace, workingSpace);

    numSamples += targets.shape[0];
    currentTotalMetric += sum(workingSpace) * -1;

    return currentTotalMetric / static_cast<float>(numSamples);
}

void CategoricalCrossEntropy::calculateDerivatives(const Tensor& targets, const Tensor& predictions,
                                                   Tensor& destination) {
    allocateWorkingSpacesDerivatives(targets, predictions);

    multiply(predictions, onesDerivatives, accumulatedSumsDerivatives);
    fill(1, destination);
    divide(destination, accumulatedSumsDerivatives, accumulatedSumsDerivatives);

    divide(targets, predictions, destination);
    multiply(destination, -1, destination);

    add(destination, accumulatedSumsDerivatives, destination);
}

void CategoricalCrossEntropy::allocateWorkingSpacesDerivatives(const Tensor& targets, const Tensor& predictions) {
    if (onesDerivatives.shape.size() != 2 || onesDerivatives.shape[0] != targets.shape[1] || onesDerivatives.shape[1] != targets.shape[1]) {
        onesDerivatives = Tensor(targets.shape[1], targets.shape[1]);
        fill(1, onesDerivatives);
    }
    onesDerivatives.move(targets.location);
    if (accumulatedSumsDerivatives.shape != targets.shape) {
        accumulatedSumsDerivatives = Tensor(targets.shape);
    }
    accumulatedSumsDerivatives.move(targets.location);
}

void CategoricalCrossEntropy::allocateWorkingSpacesLoss(const Tensor& targets, const Tensor& predictions) {
    if (onesLoss.shape.size() != 2 || onesLoss.shape[0] != targets.shape[1] || onesLoss.shape[1] != targets.shape[1]) {
        onesLoss = Tensor(targets.shape[1], targets.shape[1]);
        fill(1, onesLoss);
    }
    onesLoss.move(targets.location);
    if (accumulatedSumsLoss.shape != targets.shape) {
        accumulatedSumsLoss = Tensor(targets.shape);
    }
    accumulatedSumsLoss.move(targets.location);
    if (workingSpace.shape != targets.shape) {
        workingSpace = Tensor(targets.shape);
    }
    workingSpace.move(targets.location);
}

std::string CategoricalCrossEntropy::getShortName() const {
    return "categorical_cross_entropy";
}
