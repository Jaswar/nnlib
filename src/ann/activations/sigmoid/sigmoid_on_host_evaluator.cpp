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

DTYPE fSigmoid(DTYPE x) {
    return 1 / (1 + static_cast<DTYPE>(exp(-static_cast<double>(x))));
}

void SigmoidOnHostEvaluator::forward(const Tensor& input, Tensor& result) const {
    if (!allLocationsAreHost({input.location, result.location})) {
        throw DifferentDataLocationException();
    }

    for (size_t index = 0; index < input.size; index++) {
        result.host[index] = fSigmoid(input.host[index]);
    }
}

void SigmoidOnHostEvaluator::computeDerivatives(const Tensor& output, Tensor& result) const {
    if (!allLocationsAreHost({output.location, result.location})) {
        throw DifferentDataLocationException();
    }

    for (size_t index = 0; index < output.size; index++) {
        result.host[index] = fSigmoid(output.host[index]) * (1 - fSigmoid(output.host[index]));
    }
}

SigmoidOnHostEvaluator::~SigmoidOnHostEvaluator() = default;
