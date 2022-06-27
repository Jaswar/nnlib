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

TEST(reluActivationDevice, forward) {
    const Matrix& data = constructMatrix({{1, 0, -3}, {-4, 5, -6}}, DEVICE);
    Matrix result = Matrix(2, 3, DEVICE);

    const ReLUActivation& activation = ReLUActivation();

    activation.forward(data, result);

    result.moveToHost();

    ASSERT_MATRIX_EQ(result, {{1, 0, 0}, {0, 5, 0}});
}

TEST(reluActivationDevice, derivatives) {
    const Matrix& data = constructMatrix({{-1, 2, 0}, {4, -5, -6}}, DEVICE);
    Matrix result = Matrix(2, 3, DEVICE);

    const ReLUActivation& activation = ReLUActivation();

    activation.computeDerivatives(data, result);

    result.moveToHost();

    ASSERT_MATRIX_EQ(result, {{0, 1, 0}, {1, 0, 0}});
}

#endif
