//
// Created by Jan Warchocki on 03/03/2022.
//

#include "../../../include/activation.h"
#include <exceptions/different_data_location_exception.h>
#include <exceptions/size_mismatch_exception.h>
#include <utils/location_verifiers.h>

void Activation::forward(const Vector& input, Vector& result) const {
    if (result.n != input.n) {
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

void Activation::forward(const Matrix& input, Matrix& result) const {
    if (input.n != result.n || input.m != result.m) {
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

void Activation::computeDerivatives(const Vector& output, Vector& result) const {
    if (result.n != output.n) {
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

void Activation::computeDerivatives(const Matrix& output, Matrix& result) const {
    if (output.n != result.n || output.m != result.m) {
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
