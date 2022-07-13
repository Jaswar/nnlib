//
// Created by Jan Warchocki on 27/06/2022.
//

#include <gtest/gtest.h>
#include <matrix.h>
#include "../utils.h"
#include "../assertions.h"

TEST(matrixOperationsHost, add) {
    const Matrix& m1 = constructMatrix({{1, 2, 3}, {4, 5, 6}, {1, 2, 3}});
    const Matrix& m2 = constructMatrix({{2, 4, 8}, {16, 32, 64}, {1, 2, 3}});
    Matrix result = Matrix(3, 3);

    add(m1, m2, result);

    ASSERT_MATRIX_EQ(result, {{3, 6, 11}, {20, 37, 70}, {2, 4, 6}});
}

TEST(matrixOperationsHost, subtract) {
    const Matrix& m1 = constructMatrix({{1, 2, 3}, {4, 5, 6}, {1, 2, 3}});
    const Matrix& m2 = constructMatrix({{2, 4, 8}, {16, 32, 64}, {1, 4, 1}});
    Matrix result = Matrix(3, 3);

    subtract(m1, m2, result);

    ASSERT_MATRIX_EQ(result, {{-1, -2, -5}, {-12, -27, -58}, {0, -2, 2}});
}

TEST(matrixOperationsHost, addBroadcastVector) {
    const Matrix& m = constructMatrix({{1, 2, 3, 4, 5, 4, 3, 2, 1}, {4, 5, 6, 7, 8, 7, 6, 5, 4}});
    const Vector& v = constructVector({3, 2, 1, 0, -1, -2, -3, -4, -5});
    Matrix result = Matrix(2, 9);

    add(m, v, result);

    ASSERT_MATRIX_EQ(result, {{4, 4, 4, 4, 4, 2, 0, -2, -4}, {7, 7, 7, 7, 7, 5, 3, 1, -1}});
}

TEST(matrixOperationsHost, multiplyMatrices) {
    // 2x3 matrix
    const Matrix& m1 = constructMatrix({{1, 2, 3}, {4, 5, 6}});
    // 3x4 matrix
    const Matrix& m2 = constructMatrix({{2, 4, 8, 10}, {12, 16, 18, 20}, {22, 24, 26, 28}});
    // result should be a 2x4 matrix
    Matrix result = Matrix(2, 4);

    multiply(m1, m2, result);

    ASSERT_MATRIX_EQ(result, {{92, 108, 122, 134}, {200, 240, 278, 308}});
}

TEST(matrixOperationsHost, multiplyMatrixVector) {
    const Matrix& m = constructMatrix({{1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 4, 5, 6, 7, 8, 9, 10, 11}});
    const Vector& v = constructVector({2, 3, 4, 5, 1, 1, 1, -1, 2});
    Vector result = Vector(2);

    multiply(m, v, result);

    ASSERT_VECTOR_EQ(result, {68, 104});
}

TEST(matrixOperationsHost, multiplyMatrixConstant) {
    const Matrix& m = constructMatrix({{1, 0}, {0, 1}});
    Matrix result = Matrix(2, 2);

    multiply(m, 2, result);

    ASSERT_MATRIX_EQ(result, {{2, 0}, {0, 2}});
}

TEST(matrixOperationsHost, multiplyConstantMatrix) {
    const Matrix& m = constructMatrix({{1, 0, 2, 3, 4}, {0, 1, 3, 5, 6}});
    Matrix result = Matrix(2, 5);

    multiply(2, m, result);

    ASSERT_MATRIX_EQ(result, {{2, 0, 4, 6, 8}, {0, 2, 6, 10, 12}});
}

TEST(matrixOperationsHost, hadamardProduct) {
    const Matrix& m1 = constructMatrix({{1, 2, 3}, {4, 5, 6}, {8, 10, 12}});
    const Matrix& m2 = constructMatrix({{2, 4, 8}, {8, 4, 2}, {-2, 1, 0}});
    Matrix result = Matrix(3, 3);

    hadamard(m1, m2, result);

    ASSERT_MATRIX_EQ(result, {{2, 8, 24}, {32, 20, 12}, {-16, 10, 0}});
}

TEST(matrixOperationsHost, transpose) {
    const Matrix& m = constructMatrix({{-1, 2, -2, 2}, {7, 2, 4, 5}});
    Matrix result = Matrix(4, 2);

    transpose(m, result);

    ASSERT_MATRIX_EQ(result, {{-1, 7}, {2, 2}, {-2, 4}, {2, 5}});
}
