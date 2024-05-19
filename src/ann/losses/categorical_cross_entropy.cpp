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

sfTensor CategoricalCrossEntropy::calculateLoss(const sfTensor& targets, const sfTensor& predictions) {
    std::vector<size_t> shape = {targets->shape[1], targets->shape[1]};
    sfTensor ones = std::make_shared<Tensor<float>>(shape, targets->location);
    fill(1.0f, ones);

    sfTensor accumulatedSumsLoss = multiply(predictions, ones);

    ones = std::make_shared<Tensor<float>>(targets->shape, targets->location);
    fill(1.0f, ones);
    accumulatedSumsLoss = divide(ones, accumulatedSumsLoss);

    sfTensor workingSpace = hadamard(predictions, accumulatedSumsLoss);
    workingSpace = hadamard(targets, log(workingSpace));

    workingSpace = multiply(sum(workingSpace), -1.0f);
    return workingSpace;
}
