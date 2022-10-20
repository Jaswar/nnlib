/**
 * @file linear_on_host_evaluator.cpp
 * @brief Source file defining methods of the LinearOnHostEvaluator class.
 * @author Jan Warchocki
 * @date 28 May 2022
 */

#include "../../../../include/activation.h"
#include <exceptions/different_data_location_exception.h>
#include <utils/location_verifiers.h>

void LinearOnHostEvaluator::forward(const Tensor& input, Tensor& result) const {
    if (!allLocationsAreHost({input.location, result.location})) {
        throw DifferentDataLocationException();
    }

    for (size_t index = 0; index < input.size; index++) {
        result.data[index] = input.data[index];
    }
}

void LinearOnHostEvaluator::computeDerivatives(const Tensor& output, Tensor& result) const {
    if (!allLocationsAreHost({output.location, result.location})) {
        throw DifferentDataLocationException();
    }

    for (size_t index = 0; index < output.size; index++) {
        result.data[index] = 1;
    }
}

LinearOnHostEvaluator::~LinearOnHostEvaluator() = default;
