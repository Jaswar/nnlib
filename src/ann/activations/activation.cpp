//
// Created by Jan Warchocki on 03/03/2022.
//

#include <activation.h>
#include <exceptions/size_mismatch_exception.h>

void Activation::forward(const Vector& input, Vector& result) const {
    if (result.n != input.n) {
        throw SizeMismatchException();
    }

    this->evaluator->forward(input, result);
}

void Activation::forward(const Matrix& input, Matrix& result) const {
    if (input.n != result.n || input.m != result.m) {
        throw SizeMismatchException();
    }

    this->evaluator->forward(input, result);
}

void Activation::computeDerivatives(const Vector& output, Vector& result) const {
    if (result.n != output.n) {
        throw SizeMismatchException();
    }

    this->evaluator->computeDerivatives(output, result);
}

void Activation::computeDerivatives(const Matrix& output, Matrix& result) const {
    if (output.n != result.n || output.m != result.m) {
        throw SizeMismatchException();
    }

    this->evaluator->computeDerivatives(output, result);
}

Activation::Activation(DataLocation location) : evaluator(nullptr), location(location) {}

Activation::~Activation() {
    delete evaluator;
}
