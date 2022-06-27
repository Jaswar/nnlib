//
// Created by Jan Warchocki on 27/06/2022.
//

#include <gtest/gtest.h>
#include <matrix.h>
#include <activation.h>
#include "../utils.h"
#include "../assertions.h"

TEST(linearActivationHost, forward) {
    const Matrix& data = constructMatrix({{1, 0, -3}, {-4, 5, -6}});
    Matrix result = Matrix(2, 3);

    const LinearActivation& activation = LinearActivation();

    activation.forward(data, result);

    ASSERT_MATRIX_EQ(result, {{1, 0, -3}, {-4, 5, -6}});
}

TEST(linearActivationHost, derivatives) {
    const Matrix& data = constructMatrix({{-1, 2, 0}, {4, -5, -6}});
    Matrix result = Matrix(2, 3);

    const LinearActivation& activation = LinearActivation();

    activation.computeDerivatives(data, result);

    ASSERT_MATRIX_EQ(result, {{1, 1, 1}, {1, 1, 1}});
}