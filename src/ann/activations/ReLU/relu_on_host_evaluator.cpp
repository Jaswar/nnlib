/**
 * @file relu_on_host_evaluator.cpp
 * @brief Source file defining methods of the ReLUOnHostEvaluator class.
 * @author Jan Warchocki
 * @date 28 May 2022
 */

#include "../../../../include/activation.h"
#include <exceptions/different_data_location_exception.h>
#include <utils/location_verifiers.h>

void ReLUOnHostEvaluator::forward(const Tensor& input, Tensor& result) const {
    if (!allLocationsAreHost({input.location, result.location})) {
        throw DifferentDataLocationException();
    }

    for (size_t index = 0; index < input.size; index++) {
        if (input.host[index] <= 0) {
            result.host[index] = 0;
        } else {
            result.host[index] = input.host[index];
        }
    }
}

void ReLUOnHostEvaluator::computeDerivatives(const Tensor& output, Tensor& result) const {
    if (!allLocationsAreHost({output.location, result.location})) {
        throw DifferentDataLocationException();
    }

    for (size_t index = 0; index < output.size; index++) {
        if (output.host[index] <= 0) {
            result.host[index] = 0;
        } else {
            result.host[index] = 1;
        }
    }
}

ReLUOnHostEvaluator::~ReLUOnHostEvaluator() = default;
