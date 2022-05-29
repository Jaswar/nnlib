//
// Created by Jan Warchocki on 28/05/2022.
//

#include <utils/location_verifiers.h>
#include <exceptions/different_data_location_exception.h>
#include "../../../../include/activation.h"

void LinearOnHostEvaluator::forward(const Vector& input, Vector& result) const {
    if (!allLocationsAreHost({input.location, result.location})) {
        throw DifferentDataLocationException();
    }

    for (int i = 0; i < input.n; i++) {
        result[i] = input[i];
    }
}

void LinearOnHostEvaluator::forward(const Matrix& input, Matrix& result) const {
    if (!allLocationsAreHost({input.location, result.location})) {
        throw DifferentDataLocationException();
    }

    for (int row = 0; row < input.n; row++) {
        for (int i = 0; i < input.m; i++) {
            result(row, i) = input(row, i);
        }
    }
}

void LinearOnHostEvaluator::computeDerivatives(const Vector& output, Vector& result) const {
    if (!allLocationsAreHost({output.location, result.location})) {
        throw DifferentDataLocationException();
    }

    for (int i = 0; i < output.n; i++) {
        result[i] = 1;
    }
}

void LinearOnHostEvaluator::computeDerivatives(const Matrix& output, Matrix& result) const {
    if (!allLocationsAreHost({output.location, result.location})) {
        throw DifferentDataLocationException();
    }

    for (int row = 0; row < output.n; row++) {
        for (int i = 0; i < output.m; i++) {
            result(row, i) = 1;
        }
    }
}

LinearOnHostEvaluator::~LinearOnHostEvaluator() = default;
