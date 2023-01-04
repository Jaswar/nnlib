/**
 * @file categorical_cross_entropy.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 03 January 2023
 */

#include <loss.h>

float CategoricalCrossEntropy::calculateLoss(const Tensor& targets, const Tensor& predictions) {
    return 0;
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