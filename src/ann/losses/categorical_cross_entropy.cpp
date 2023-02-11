/**
 * @file categorical_cross_entropy.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 03 January 2023
 */

#include <loss.h>

float CategoricalCrossEntropy::calculateLoss(const Tensor& targets, const Tensor& predictions) {
    allocateWorkingSpace(targets, predictions);
    allocateOnes(targets, predictions);
    allocateAccumulatedSums(targets, predictions);

    multiply(predictions, ones, accumulatedSums);
    fill(1, workingSpace);
    divide(workingSpace, accumulatedSums, accumulatedSums);

    hadamard(predictions, accumulatedSums, workingSpace);
    log(workingSpace, workingSpace);
    hadamard(targets, workingSpace, workingSpace);

    numSamples += targets.shape[0];
    currentTotalLoss += sum(workingSpace) * -1;

    return currentTotalLoss / static_cast<float>(numSamples);
}

void CategoricalCrossEntropy::calculateDerivatives(const Tensor& targets, const Tensor& predictions,
                                                   Tensor& destination) {
    allocateOnes(targets, predictions);
    allocateAccumulatedSums(targets, predictions);

    multiply(predictions, ones, accumulatedSums);
    fill(1, destination);
    divide(destination, accumulatedSums, accumulatedSums);

    divide(targets, predictions, destination);
    multiply(destination, -1, destination);

    add(destination, accumulatedSums, destination);
}

void CategoricalCrossEntropy::allocateWorkingSpace(const Tensor& targets, const Tensor& predictions) {
    if (workingSpace.shape != targets.shape) {
        workingSpace = Tensor(targets.shape);
    }
    if (workingSpace.location != targets.location) {
        workingSpace.move(targets.location);
    }
}

void CategoricalCrossEntropy::allocateOnes(const Tensor& targets, const Tensor& predictions) {
    if (ones.shape.size() != 2 || ones.shape[0] != targets.shape[1] || ones.shape[1] != targets.shape[1]) {
        ones = Tensor(targets.shape[1], targets.shape[1]);
        fill(1, ones);
    }
    if (ones.location != targets.location) {
        ones.move(targets.location);
    }
}

void CategoricalCrossEntropy::allocateAccumulatedSums(const Tensor& targets, const Tensor& predictions) {
    if (accumulatedSums.shape != targets.shape) {
        accumulatedSums = Tensor(targets.shape);
    }
    if (accumulatedSums.location != targets.location) {
        accumulatedSums.move(targets.location);
    }
}