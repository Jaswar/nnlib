//
// Created by Jan Warchocki on 27/06/2022.
//

#include <gtest/gtest.h>
#include <matrix.h>
#include "../utils.h"
#include "../assertions.h"

TEST(matrix_operations_host, add) {
    const Matrix& m1 = constructMatrix({{1, 2, 3}, {4, 5, 6}});
    const Matrix& m2 = constructMatrix({{2, 4, 8}, {16, 32, 64}});
    Matrix result = Matrix(2, 3);

    add(m1, m2, result);

    ASSERT_MATRIX(result, {{3, 6, 11}, {20, 37, 70}});
}

TEST(matrix_operations_host, subtract) {
    const Matrix& m1 = constructMatrix({{1, 2, 3}, {4, 5, 6}});
    const Matrix& m2 = constructMatrix({{2, 4, 8}, {16, 32, 64}});
    Matrix result = Matrix(2, 3);

    subtract(m1, m2, result);

    ASSERT_MATRIX(result, {{-1, -2, -5}, {-12, -27, -58}});
}

TEST(matrix_operations_host, add_broadcast_vector) {
    const Matrix& m = constructMatrix({{1, 2, 3}, {4, 5, 6}});
    const Vector& v = constructVector({3, 2, 1});
    Matrix result = Matrix(2, 3);

    add(m, v, result);

    ASSERT_MATRIX(result, {{4, 4, 4}, {7, 7, 7}});
}

TEST(matrix_operations_host, multiply_matrices) {
    // 2x3 matrix
    const Matrix& m1 = constructMatrix({{1, 2, 3}, {4, 5, 6}});
    // 3x4 matrix
    const Matrix& m2 = constructMatrix({{2, 4, 8, 10}, {12, 16, 18, 20}, {22, 24, 26, 28}});
    // result should be a 2x4 matrix
    Matrix result = Matrix(2, 4);

    multiply(m1, m2, result);

    ASSERT_MATRIX(result, {{92, 108, 122, 134}, {200, 240, 278, 308}});
}

TEST(matrix_operations_host, multiply_matrix_vector) {
    const Matrix& m = constructMatrix({{1, 2}, {3, 4}});
    const Vector& v = constructVector({2, 3});
    Vector result = Vector(2);

    multiply(m, v, result);

    ASSERT_VECTOR(result, {8, 18});
}

TEST(matrix_operations_host, multiply_matrix_constant) {
    const Matrix& m = constructMatrix({{1, 0}, {0, 1}});
    Matrix result = Matrix(2, 2);

    multiply(m, 2, result);

    ASSERT_MATRIX(result, {{2, 0}, {0, 2}});
}

TEST(matrix_operations_host, multiply_constant_matrix) {
    const Matrix& m = constructMatrix({{1, 0}, {0, 1}});
    Matrix result = Matrix(2, 2);

    multiply(2, m, result);

    ASSERT_MATRIX(result, {{2, 0}, {0, 2}});
}

TEST(matrix_operations_host, hadamard_product) {
    const Matrix& m1 = constructMatrix({{1, 2, 3}, {4, 5, 6}});
    const Matrix& m2 = constructMatrix({{2, 4, 8}, {8, 4, 2}});
    Matrix result = Matrix(2, 3);

    hadamard(m1, m2, result);

    ASSERT_MATRIX(result, {{2, 8, 24}, {32, 20, 12}});
}

TEST(matrix_operations_host, transpose) {
    const Matrix& m = constructMatrix({{-1, 2, -2, 2}, {7, 2, 4, 5}});
    Matrix result = Matrix(4, 2);

    transpose(m, result);

    ASSERT_MATRIX(result, {{-1, 7}, {2, 2}, {-2, 4}, {2, 5}});
}
