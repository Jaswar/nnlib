/**
 * @file binary_cross_entropy.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 24 December 2022
 */

#include <loss.h>
#include <exceptions/unsupported_operation_exception.h>

void checkValidShape(const Tensor& targets, const Tensor& predictions) {
    if (targets.shape.size() != 2 || targets.shape[1] != 1) {
        throw UnsupportedOperationException();
    }
    if (predictions.shape.size() != 2 || predictions.shape[1] != 1) {
        throw UnsupportedOperationException();
    }
}

float BinaryCrossEntropy::calculateLoss(const Tensor& targets, const Tensor& predictions) {
    checkValidShape(targets, predictions);

    return 0;
}

void BinaryCrossEntropy::calculateDerivatives(const Tensor& targets, const Tensor& predictions, Tensor& destination) {
    checkValidShape(targets, predictions);

    if (ones.shape != targets.shape) {
        ones = Tensor(targets.shape);
        fill(1, ones);
    }
    if (workingSpace.shape != targets.shape) {
        workingSpace = Tensor(targets.shape);
    }
    if (ones.location != targets.location) {
        ones.move(targets.location);
    }
    if (workingSpace.location != targets.location) {
        workingSpace.move(targets.location);
    }

    // Calculate the nominator
    subtract(predictions, targets, destination);

    // Calculate the denominator
    subtract(ones, predictions, workingSpace);
    hadamard(predictions, workingSpace, workingSpace);

    // Calculate the fraction
    divide(destination, workingSpace, destination);
}