/**
 * @file binary_cross_entropy.cpp
 * @brief Source file defining the methods of the BinaryCrossEntropy class.
 * @author Jan Warchocki
 * @date 24 December 2022
 */

#include <exceptions/unsupported_operation_exception.h>
#include <loss.h>

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

    allocateWorkingSpacesLoss(targets, predictions);

    subtract(onesLoss, targets, workingSpace);
    subtract(onesLoss, predictions, workingSpace2);
    log(workingSpace2, workingSpace2);
    hadamard(workingSpace, workingSpace2, workingSpace2);

    log(predictions, workingSpace);
    hadamard(targets, workingSpace, workingSpace);

    add(workingSpace, workingSpace2, workingSpace);

    numSamples += targets.shape[0];
    currentTotalMetric += sum(workingSpace) * -1;

    return currentTotalMetric / static_cast<float>(numSamples);
}

void BinaryCrossEntropy::calculateDerivatives(const Tensor& targets, const Tensor& predictions, Tensor& destination) {
    checkValidShape(targets, predictions);

    allocateWorkingSpacesDerivatives(targets, predictions);

    // Calculate the nominator
    subtract(predictions, targets, destination);

    // Calculate the denominator
    subtract(onesDerivatives, predictions, workingSpace3);
    hadamard(predictions, workingSpace3, workingSpace3);

    // Calculate the fraction
    divide(destination, workingSpace3, destination);
}

void BinaryCrossEntropy::allocateWorkingSpacesDerivatives(const Tensor& targets, const Tensor& predictions) {
    if (workingSpace3.shape != targets.shape) {
        workingSpace3 = Tensor(targets.shape);
    }
    workingSpace3.move(targets.location);
    if (onesDerivatives.shape != targets.shape) {
        onesDerivatives = Tensor(targets.shape);
        fill(1, onesDerivatives);
    }
    onesDerivatives.move(targets.location);
}

void BinaryCrossEntropy::allocateWorkingSpacesLoss(const Tensor& targets, const Tensor& predictions) {
    if (workingSpace.shape != targets.shape) {
        workingSpace = Tensor(targets.shape);
    }
    workingSpace.move(targets.location);
    if (workingSpace2.shape != targets.shape) {
        workingSpace2 = Tensor(targets.shape);
    }
    workingSpace2.move(targets.location);
    if (onesLoss.shape != targets.shape) {
        onesLoss = Tensor(targets.shape);
        fill(1, onesLoss);
    }
    onesLoss.move(targets.location);
}

std::string BinaryCrossEntropy::getShortName() const {
    return "binary_cross_entropy";
}
