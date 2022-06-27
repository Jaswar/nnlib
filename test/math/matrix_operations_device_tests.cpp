//
// Created by Jan Warchocki on 27/06/2022.
//

#include <gtest/gtest.h>
#include <matrix.h>
#include "../utils.h"
#include "../assertions.h"
#include <verify.cuh>

#ifdef HAS_CUDA

TEST(matrix_operations_device, add) {
    const Matrix& m1 = constructMatrix({{1, 2, 3}, {4, 5, 6}}, DEVICE);
    const Matrix& m2 = constructMatrix({{2, 4, 8}, {16, 32, 64}}, DEVICE);
    Matrix result = Matrix(2, 3, DEVICE);

    add(m1, m2, result);

    result.moveToHost();

    ASSERT_MATRIX_EQ(result, {{3, 6, 11}, {20, 37, 70}});
}

TEST(matrix_operations_device, subtract) {
    const Matrix& m1 = constructMatrix({{1, 2, 3}, {4, 5, 6}}, DEVICE);
    const Matrix& m2 = constructMatrix({{2, 4, 8}, {16, 32, 64}}, DEVICE);
    Matrix result = Matrix(2, 3, DEVICE);

    subtract(m1, m2, result);

    result.moveToHost();

    ASSERT_MATRIX_EQ(result, {{-1, -2, -5}, {-12, -27, -58}});
}

TEST(matrix_operations_device, add_broadcast_vector) {
    const Matrix& m = constructMatrix({{1, 2, 3}, {4, 5, 6}}, DEVICE);
    const Vector& v = constructVector({3, 2, 1}, DEVICE);
    Matrix result = Matrix(2, 3, DEVICE);

    add(m, v, result);

    result.moveToHost();

    ASSERT_MATRIX_EQ(result, {{4, 4, 4}, {7, 7, 7}});
}

TEST(matrix_operations_device, multiply_matrices) {
    // 2x3 matrix
    const Matrix& m1 = constructMatrix({{1, 2, 3}, {4, 5, 6}}, DEVICE);
    // 3x4 matrix
    const Matrix& m2 = constructMatrix({{2, 4, 8, 10}, {12, 16, 18, 20}, {22, 24, 26, 28}}, DEVICE);
    // result should be a 2x4 matrix
    Matrix result = Matrix(2, 4, DEVICE);

    multiply(m1, m2, result);

    result.moveToHost();

    ASSERT_MATRIX_EQ(result, {{92, 108, 122, 134}, {200, 240, 278, 308}});
}

TEST(matrix_operations_device, multiply_matrix_vector) {
    const Matrix& m = constructMatrix({{1, 2}, {3, 4}}, DEVICE);
    const Vector& v = constructVector({2, 3}, DEVICE);
    Vector result = Vector(2, DEVICE);

    multiply(m, v, result);

    result.moveToHost();

    ASSERT_VECTOR_EQ(result, {8, 18});
}

TEST(matrix_operations_device, multiply_matrix_constant) {
    const Matrix& m = constructMatrix({{1, 0}, {0, 1}}, DEVICE);
    Matrix result = Matrix(2, 2, DEVICE);

    multiply(m, 2, result);

    result.moveToHost();

    ASSERT_MATRIX_EQ(result, {{2, 0}, {0, 2}});
}

TEST(matrix_operations_device, multiply_constant_matrix) {
    const Matrix& m = constructMatrix({{1, 0}, {0, 1}}, DEVICE);
    Matrix result = Matrix(2, 2, DEVICE);

    multiply(2, m, result);

    result.moveToHost();

    ASSERT_MATRIX_EQ(result, {{2, 0}, {0, 2}});
}

TEST(matrix_operations_device, hadamard_product) {
    const Matrix& m1 = constructMatrix({{1, 2, 3}, {4, 5, 6}}, DEVICE);
    const Matrix& m2 = constructMatrix({{2, 4, 8}, {8, 4, 2}}, DEVICE);
    Matrix result = Matrix(2, 3, DEVICE);

    hadamard(m1, m2, result);

    result.moveToHost();

    ASSERT_MATRIX_EQ(result, {{2, 8, 24}, {32, 20, 12}});
}

TEST(matrix_operations_device, transpose) {
    const Matrix& m = constructMatrix({{-1, 2, -2, 2}, {7, 2, 4, 5}}, DEVICE);
    Matrix result = Matrix(4, 2, DEVICE);

    transpose(m, result);

    result.moveToHost();

    ASSERT_MATRIX_EQ(result, {{-1, 7}, {2, 2}, {-2, 4}, {2, 5}});
}

#endif
