/**
 * @file activation.cpp
 * @brief Source file defining methods of the Activation class.
 * @author Jan Warchocki
 * @date 03 March 2022
 */

#include "../../../include/activation.h"
#include <exceptions/different_data_location_exception.h>
#include <exceptions/size_mismatch_exception.h>
#include <utils/location_verifiers.h>

void Activation::forward(const Tensor& input, Tensor& result) const {
    if (input.shape != result.shape) {
        throw SizeMismatchException();
    }

    if (allLocationsAreHost({input.location, result.location})) {
        this->hostEvaluator->forward(input, result);
    } else if (allLocationsAreDevice({input.location, result.location})) {
        this->deviceEvaluator->forward(input, result);
    } else {
        throw DifferentDataLocationException();
    }
}

void Activation::computeDerivatives(const Tensor& output, Tensor& result) const {
    if (output.shape != result.shape) {
        throw SizeMismatchException();
    }

    if (allLocationsAreHost({output.location, result.location})) {
        this->hostEvaluator->computeDerivatives(output, result);
    } else if (allLocationsAreDevice({output.location, result.location})) {
        this->deviceEvaluator->computeDerivatives(output, result);
    } else {
        throw DifferentDataLocationException();
    }
}

Activation::Activation(ActivationEvaluator* hostEvaluator, ActivationEvaluator* deviceEvaluator)
    : hostEvaluator(hostEvaluator), deviceEvaluator(deviceEvaluator) {
}

Activation::~Activation() {
    delete hostEvaluator;
    delete deviceEvaluator;
}
