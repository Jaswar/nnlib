/**
 * @file sigmoid_on_host_evaluator.cpp
 * @brief Source file defining methods of the SigmoidOnHostEvaluator class.
 * @author Jan Warchocki
 * @date 28 May 2022
 */


#include "../../../../include/activation.h"
#include <cmath>
#include <exceptions/different_data_location_exception.h>
#include <utils/location_verifiers.h>

float fSigmoid(float x) {
    return 1 / (1 + static_cast<float>(exp(-static_cast<double>(x))));
}

void SigmoidOnHostEvaluator::forward(const Tensor& input, Tensor& result) const {
    if (!allLocationsAreHost({input.location, result.location})) {
        throw DifferentDataLocationException();
    }

    for (size_t index = 0; index < input.size; index++) {
        result.data[index] = fSigmoid(input.data[index]);
    }
}

void SigmoidOnHostEvaluator::computeDerivatives(const Tensor& output, Tensor& result) const {
    if (!allLocationsAreHost({output.location, result.location})) {
        throw DifferentDataLocationException();
    }

    for (size_t index = 0; index < output.size; index++) {
        result.data[index] = fSigmoid(output.data[index]) * (1 - fSigmoid(output.data[index]));
    }
}

SigmoidOnHostEvaluator::~SigmoidOnHostEvaluator() = default;
