//
// Created by Jan Warchocki on 27/06/2022.
//

#include <gtest/gtest.h>
#include <matrix.h>
#include <activation.h>
#include "../utils.h"
#include "../assertions.h"

TEST(sigmoidActivationHost, forward) {
    const Matrix& data = constructMatrix({{1, 0, -3}, {-4, 5, -6}});
    Matrix result = Matrix(2, 3);

    const SigmoidActivation& activation = SigmoidActivation();

    activation.forward(data, result);

    ASSERT_MATRIX_CLOSE(result, {{0.73105, 0.5, 0.04742}, {0.01798, 0.99330, 0.00247}});
}

TEST(sigmoidActivationHost, derivatives) {
    const Matrix& data = constructMatrix({{-1, 2, 0}, {4, -5, -6}});
    Matrix result = Matrix(2, 3);

    const SigmoidActivation& activation = SigmoidActivation();

    activation.computeDerivatives(data, result);

    ASSERT_MATRIX_CLOSE(result, {{0.19661, 0.10499, 0.25}, {0.01766, 0.00664, 0.00246}});
}