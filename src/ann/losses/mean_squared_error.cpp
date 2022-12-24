/**
 * @file mean_squared_error.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 23 December 2022
 */

#include <loss.h>

void MeanSquaredError::calculateDerivatives(const Tensor& targets, const Tensor& predictions, Tensor& destination) {
    subtract(predictions, targets, destination);
    multiply(destination, 2.0f / static_cast<float>(predictions.shape[predictions.shape.size() - 1]), destination);
}

float MeanSquaredError::calculateLoss(const Tensor& targets, const Tensor& predictions) {
    if (workingSpace.shape != targets.shape) {
        workingSpace = Tensor(targets.shape);
    }
    if (targets.location != workingSpace.location) {
        workingSpace.move(targets.location);
    }

    size_t numSamples = targets.shape[targets.shape.size() - 1];

    subtract(targets, predictions, workingSpace);
    hadamard(workingSpace, workingSpace, workingSpace);
    multiply(workingSpace, 1.0f / static_cast<float>(numSamples), workingSpace);

    return sum(workingSpace);
}
