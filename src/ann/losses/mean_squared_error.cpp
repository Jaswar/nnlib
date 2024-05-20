/**
 * @file mean_squared_error.cpp
 * @brief Source file defining the methods of the MeanSquaredError class.
 * @author Jan Warchocki
 * @date 23 December 2022
 */

#include <loss.h>
#include <functions.h>

std::string MeanSquaredError::getShortName() const {
    return "mean_squared_error";
}

sfTensor MeanSquaredError::calculateLoss(const sfTensor& targets, const sfTensor& predictions) {
    size_t numOutputs = targets->shape[targets->shape.size() - 1];

    sfTensor difference = subtract(predictions, targets);
    sfTensor loss = hadamard(difference, difference);
    loss = multiply(sum(loss), 1.0f / static_cast<float>(numOutputs));

    return loss;
}
