//
// Created by Jan Warchocki on 27/06/2022.
//

#include <gtest/gtest.h>
#include <matrix.h>
#include "utils.h"

TEST(matrix_operations_device, add) {
    const Matrix& m1 = constructMatrix({{1, 2, 3}, {4, 5, 6}});
    const Matrix& m2 = constructMatrix({{2, 4, 8}, {16, 32, 64}});
    Matrix result = Matrix(2, 3, HOST);

    add(m1, m2, result);

    assertEqual(result, {{3, 6, 11}, {20, 37, 70}});
}
