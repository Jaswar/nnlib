/**
 * @file categorical_cross_entropy.cpp
 * @brief Source file defining the methods of the CategoricalCrossEntropy class.
 * @author Jan Warchocki
 * @date 03 January 2023
 */

#include <loss.h>

std::string CategoricalCrossEntropy::getShortName() const {
    return "categorical_cross_entropy";
}

sTensor CategoricalCrossEntropy::calculateLoss(const sTensor& targets, const sTensor& predictions) {
    std::vector<size_t> shape = {targets->shape[1], targets->shape[1]};
    sTensor ones = std::make_shared<Tensor>(shape, targets->location);
    fill(1.0f, ones);

    sTensor accumulatedSumsLoss = multiply(predictions, ones);

    ones = std::make_shared<Tensor>(targets->shape, targets->location);
    fill(1.0f, ones);
    accumulatedSumsLoss = divide(ones, accumulatedSumsLoss);

    sTensor workingSpace = hadamard(predictions, accumulatedSumsLoss);
    workingSpace = hadamard(targets, log(workingSpace));

    workingSpace = multiply(sum(workingSpace), -1.0f);
    return workingSpace;
}
