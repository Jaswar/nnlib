/**
 * @file mean_squared_error.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 23 December 2022
 */

#include <loss.h>

void MeanSquaredError::calculateDerivatives(const Tensor& targets, const Tensor& predictions,
                                            Tensor& destination) const {
    subtract(predictions, targets, destination);
    multiply(destination, 1.0f / static_cast<float>(predictions.shape[predictions.shape.size() - 1]), destination);
}

// TODO: implement
float MeanSquaredError::calculateLoss(const Tensor& targets, const Tensor& predictions) const {
    DataLocation locationTargets = targets.location;
    DataLocation predictionsTargets = predictions.location;

    return 0;
}
