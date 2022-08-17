/**
 * @file relu_on_host_evaluator.cpp
 * @brief Source file defining methods of the ReLUOnHostEvaluator class.
 * @author Jan Warchocki
 * @date 28 May 2022
 */

#include "../../../../include/activation.h"
#include <exceptions/different_data_location_exception.h>
#include <utils/location_verifiers.h>

void ReLUOnHostEvaluator::forward(const Vector& input, Vector& result) const {
    if (!allLocationsAreHost({input.location, result.location})) {
        throw DifferentDataLocationException();
    }

    for (int i = 0; i < input.n; i++) {
        if (input[i] <= 0) {
            result[i] = 0;
        } else {
            result[i] = input[i];
        }
    }
}

void ReLUOnHostEvaluator::forward(const Matrix& input, Matrix& result) const {
    if (!allLocationsAreHost({input.location, result.location})) {
        throw DifferentDataLocationException();
    }

    for (int row = 0; row < input.n; row++) {
        for (int i = 0; i < input.m; i++) {
            if (input(row, i) <= 0) {
                result(row, i) = 0;
            } else {
                result(row, i) = input(row, i);
            }
        }
    }
}

void ReLUOnHostEvaluator::computeDerivatives(const Vector& output, Vector& result) const {
    if (!allLocationsAreHost({output.location, result.location})) {
        throw DifferentDataLocationException();
    }

    for (int i = 0; i < output.n; i++) {
        if (output[i] <= 0) {
            result[i] = 0;
        } else {
            result[i] = 1;
        }
    }
}

void ReLUOnHostEvaluator::computeDerivatives(const Matrix& output, Matrix& result) const {
    if (!allLocationsAreHost({output.location, result.location})) {
        throw DifferentDataLocationException();
    }

    for (int row = 0; row < output.n; row++) {
        for (int i = 0; i < output.m; i++) {
            if (output(row, i) <= 0) {
                result(row, i) = 0;
            } else {
                result(row, i) = 1;
            }
        }
    }
}

ReLUOnHostEvaluator::~ReLUOnHostEvaluator() = default;
