/**
 * @file mean_squared_error.cpp
 * @brief Source file defining the methods of the MeanSquaredError class.
 * @author Jan Warchocki
 * @date 23 December 2022
 */

#include <loss.h>

Tensor MeanSquaredError::calculateDerivatives(const Tensor& targets, const Tensor& predictions) {
    Tensor result = subtract(predictions, targets);
    result = multiply(result, 2.0f / static_cast<float>(predictions.shape[predictions.shape.size() - 1]));
    return result;
}

float MeanSquaredError::calculateLoss(const Tensor& targets, const Tensor& predictions) {
    size_t numOutputs = targets.shape[targets.shape.size() - 1];

    Tensor loss = subtract(predictions, targets);
    loss = hadamard(loss, loss);

    numSamples += targets.shape[0];
    currentTotalMetric += sum(loss) / static_cast<float>(numOutputs);

    return currentTotalMetric / static_cast<float>(numSamples);
}

std::string MeanSquaredError::getShortName() const {
    return "mean_squared_error";
}

sTensor MeanSquaredError::calculateLoss(const sTensor& targets, const sTensor& predictions) {
    size_t numOutputs = targets->shape[targets->shape.size() - 1];

    sTensor difference = subtract(predictions, targets);
    sTensor loss = hadamard(difference, difference);
    loss = multiply(sum(loss), 1.0f / static_cast<float>(numOutputs));

    return loss;
}
