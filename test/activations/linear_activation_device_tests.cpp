//
// Created by Jan Warchocki on 27/06/2022.
//

#include <gtest/gtest.h>
#include <matrix.h>
#include <activation.h>
#include "../utils.h"
#include "../assertions.h"
#include <verify.cuh>

#ifdef HAS_CUDA

TEST(linearActivationDevice, forward) {
    const Matrix& data = constructMatrix({{1, 0, -3}, {-4, 5, -6}}, DEVICE);
    Matrix result = Matrix(2, 3, DEVICE);

    const LinearActivation& activation = LinearActivation();

    activation.forward(data, result);

    result.moveToHost();

    ASSERT_MATRIX_EQ(result, {{1, 0, -3}, {-4, 5, -6}});
}

TEST(linearActivationDevice, derivatives) {
    const Matrix& data = constructMatrix({{-1, 2, 0}, {4, -5, -6}}, DEVICE);
    Matrix result = Matrix(2, 3, DEVICE);

    const LinearActivation& activation = LinearActivation();

    activation.computeDerivatives(data, result);

    result.moveToHost();

    ASSERT_MATRIX_EQ(result, {{1, 1, 1}, {1, 1, 1}});
}

#endif
