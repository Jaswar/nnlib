//
// Created by Jan Warchocki on 29/05/2022.
//


#include <activation.h>
#include <utils/location_verifiers.h>
#include <exceptions/different_data_location_exception.h>

DTYPE fSigmoid(DTYPE x) {
    return 1 / (1 + exp(-x));
}

void SigmoidOnHostEvaluator::forward(const Vector& input, Vector& result) const {
    if (!allLocationsAreHost({input.location, result.location})) {
        throw DifferentDataLocationException();
    }

    for (int i = 0; i < input.n; i++) {
        result[i] = fSigmoid(input[i]);
    }
}

void SigmoidOnHostEvaluator::forward(const Matrix& input, Matrix& result) const {
    if (!allLocationsAreHost({input.location, result.location})) {
        throw DifferentDataLocationException();
    }

    for (int row = 0; row < input.n; row++) {
        for (int i = 0; i < input.m; i++) {
            result(row, i) = fSigmoid(input(row, i));
        }
    }
}

void SigmoidOnHostEvaluator::computeDerivatives(const Vector& output, Vector& result) const {
    if (!allLocationsAreHost({output.location, result.location})) {
        throw DifferentDataLocationException();
    }

    for (int i = 0; i < output.n; i++) {
        result[i] = fSigmoid(output[i]) * (1 - fSigmoid(output[i]));
    }
}

void SigmoidOnHostEvaluator::computeDerivatives(const Matrix& output, Matrix& result) const {
    if (!allLocationsAreHost({output.location, result.location})) {
        throw DifferentDataLocationException();
    }

    for (int row = 0; row < output.n; row++) {
        for (int i = 0; i < output.m; i++) {
            result(row, i) = fSigmoid(output(row, i)) * (1 - fSigmoid(output(row, i)));
        }
    }
}

SigmoidOnHostEvaluator::~SigmoidOnHostEvaluator() = default;