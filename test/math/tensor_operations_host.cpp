//
// Created by Jan Warchocki on 26/08/2022.
//

#include <gtest/gtest.h>
#include <verify.cuh>
#include "../assertions.h"
#include "../test_utils.h"
#include <tensor.h>


TEST(tensor_operations_host, fill) {
    Tensor t1 = Tensor::construct1d({2, 3, 4, 5, 6, 7, 8, 9, 0});

    fill(2, t1);

    ASSERT_TENSOR_EQ_1D(t1, {2, 2, 2, 2, 2, 2, 2, 2, 2});
}

TEST(tensor_operations_host, add) {
    Tensor t1 = Tensor::construct1d({2, 3, 4, 5, 6, 7, 8, 9, 0});
    Tensor t2 = Tensor::construct1d({5, 3, 5, 9, 9, 10, 22, 11, 2});
    Tensor result = Tensor(t1.shape[0]);

    add(t1, t2, result);

    ASSERT_TENSOR_EQ_1D(result, {7, 6, 9, 14, 15, 17, 30, 20, 2});
}

TEST(tensor_operations_host, add_broadcast) {
    Tensor t1 = Tensor::construct2d({{2, 3, 4}, {5, 6, 7}, {8, 9, 0}});
    Tensor t2 = Tensor::construct1d({5, 3, -5});
    Tensor result = Tensor(t1.shape[0], t1.shape[1]);

    add(t1, t2, result);

    ASSERT_TENSOR_EQ_2D(result, {{7, 6, -1}, {10, 9, 2}, {13, 12, -5}});
}

TEST(tensor_operations_host, subtract) {
    Tensor t1 = Tensor::construct1d({2, 3, 4, 5, 6, 7, 8, 9, 0});
    Tensor t2 = Tensor::construct1d({5, 3, 5, 1, 2, 10, -4, 11, 2});
    Tensor result = Tensor(t1.shape[0]);

    subtract(t1, t2, result);

    ASSERT_TENSOR_EQ_1D(result, {-3, 0, -1, 4, 4, -3, 12, -2, -2});
}

TEST(tensor_operations_host, hadamard) {
    Tensor t1 = Tensor::construct1d({-1, 4, 0.5, 1, 2, 0, 12, 11, -2.5});
    Tensor t2 = Tensor::construct1d({7, -2, 2, 3, 7, 10, -2, -1, -2});
    Tensor result = Tensor(t1.shape[0]);

    hadamard(t1, t2, result);

    ASSERT_TENSOR_EQ_1D(result, {-7, -8, 1, 3, 14, 0, -24, -11, 5});
}

//TEST(tensor_operations_host, log) {
//    Tensor t1 = Tensor::construct1d({2.71828182, 0.01}); //-0.69314718246459961
//    Tensor result = Tensor(t1.shape[0]);
//
//    log(t1, result);
//
//    ASSERT_TENSOR_CLOSE_1D(result, {1, 1});
//}

TEST(tensor_operations_host, multiply_constant) {
    Tensor t1 = Tensor::construct1d({2, 3, -4, -5, -6, 7, 8, 9, 0});
    Tensor result = Tensor(t1.shape[0]);

    multiply(t1, -2, result);

    ASSERT_TENSOR_EQ_1D(result, {-4, -6, 8, 10, 12, -14, -16, -18, 0});
}

TEST(tensor_operations_host, multiply_matrix_vector) {
    Tensor matrix = initializeRandom(10, 12);
    Tensor vector = initializeRandom(12);
    Tensor result = Tensor(10);
    Tensor expected = Tensor(10);

    for (size_t i = 0; i < matrix.shape[0]; i++) {
        expected(i) = 0;
        for (size_t j = 0; j < matrix.shape[1]; j++) {
            expected(i) += matrix(i, j) * vector(j);
        }
    }

    multiply(matrix, vector, result);

    ASSERT_TENSOR_CLOSE(result, expected);
}

TEST(tensor_operations_host, multiply_matrix_matrix) {
    Tensor m1 = initializeRandom(10, 12);
    Tensor m2 = initializeRandom(12, 15);
    Tensor result = Tensor(10, 15);
    Tensor expected = Tensor(10, 15);

    for (size_t i = 0; i < expected.shape[0]; i++) {
        for (size_t j = 0; j < expected.shape[1]; j++) {
            expected(i, j) = 0;
            for (size_t k = 0; k < m1.shape[1]; k++) {
                expected(i, j) += m1(i, k) * m2(k, j);
            }
        }
    }

    multiply(m1, m2, result);

    ASSERT_TENSOR_CLOSE(result, expected);
}

TEST(tensor_operations_host, transpose) {
    Tensor t1 = Tensor::construct2d({{1, 2, 3, 4}, {4, 5, 6, 7}, {7, 8, 9, 10}});
    Tensor result = Tensor(t1.shape[1], t1.shape[0]);

    transpose(t1, result);

    ASSERT_TENSOR_EQ_2D(result, {{1, 4, 7}, {2, 5, 8}, {3, 6, 9}, {4, 7, 10}});
}
