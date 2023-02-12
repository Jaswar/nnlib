/**
 * @file binary_cross_entropy.cpp
 * @brief Source file defining the methods of the BinaryCrossEntropy class.
 * @author Jan Warchocki
 * @date 24 December 2022
 */

#include <loss.h>
#include <exceptions/unsupported_operation_exception.h>

/**
 * @brief Check if the shape is valid for Binary Cross Entropy.
 *
 * The shape should be (n, 1) for both @p targets and @p predictions.
 *
 * @param targets The expected outputs of the network.
 * @param predictions The actual outputs of the network.
 */
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

    allocateOnes(targets, predictions);
    allocateWorkingSpaces(targets, predictions);

    subtract(ones, targets, workingSpace);
    subtract(ones, predictions, workingSpace2);
    log(workingSpace2, workingSpace2);
    hadamard(workingSpace, workingSpace2, workingSpace2);

    log(predictions, workingSpace);
    hadamard(targets, workingSpace, workingSpace);

    add(workingSpace, workingSpace2, workingSpace);

    numSamples += targets.shape[0];
    currentTotalLoss += sum(workingSpace) * -1;

    return currentTotalLoss / static_cast<float>(numSamples);
}

void BinaryCrossEntropy::calculateDerivatives(const Tensor& targets, const Tensor& predictions, Tensor& destination) {
    checkValidShape(targets, predictions);

    allocateOnes(targets, predictions);
    allocateWorkingSpaces(targets, predictions);

    // Calculate the nominator
    subtract(predictions, targets, destination);

    // Calculate the denominator
    subtract(ones, predictions, workingSpace);
    hadamard(predictions, workingSpace, workingSpace);

    // Calculate the fraction
    divide(destination, workingSpace, destination);
}

void BinaryCrossEntropy::allocateOnes(const Tensor& targets, const Tensor& predictions) {
    if (ones.shape != targets.shape) {
        ones = Tensor(targets.shape);
        fill(1, ones);
    }
    if (ones.location != targets.location) {
        ones.move(targets.location);
    }
}

void BinaryCrossEntropy::allocateWorkingSpaces(const Tensor& targets, const Tensor& predictions) {
    if (workingSpace.shape != targets.shape) {
        workingSpace = Tensor(targets.shape);
    }
    if (workingSpace2.shape != targets.shape) {
        workingSpace2 = Tensor(targets.shape);
    }
    if (workingSpace.location != targets.location) {
        workingSpace.move(targets.location);
    }
    if (workingSpace2.location != targets.location) {
        workingSpace2.move(targets.location);
    }
}
