//
// Created by Jan Warchocki on 26/08/2022.
//

#include <gtest/gtest.h>
#include <verify.cuh>
#include "../assertions.h"
#include <tensor.h>

TEST(tensor_operations_host, fill) {
    Tensor t1 = Tensor::construct1d({2, 3, 4, 5, 6, 7, 8, 9, 0});

    fill(2, t1);

    ASSERT_EQ_1D(t1, {2, 2, 2, 2, 2, 2, 2, 2, 2});
}

TEST(tensor_operations_host, add) {
    Tensor t1 = Tensor::construct1d({2, 3, 4, 5, 6, 7, 8, 9, 0});
    Tensor t2 = Tensor::construct1d({5, 3, 5, 9, 9, 10, 22, 11, 2});
    Tensor result = Tensor(t1.shape[0]);

    add(t1, t2, result);

    ASSERT_EQ_1D(result, {7, 6, 9, 14, 15, 17, 30, 20, 2});
}

TEST(tensor_operations_host, subtract) {
    Tensor t1 = Tensor::construct1d({2, 3, 4, 5, 6, 7, 8, 9, 0});
    Tensor t2 = Tensor::construct1d({5, 3, 5, 1, 2, 10, -4, 11, 2});
    Tensor result = Tensor(t1.shape[0]);

    subtract(t1, t2, result);

    ASSERT_EQ_1D(result, {-3, 0, -1, 4, 4, -3, 12, -2, -2});
}

TEST(tensor_operations_host, hadamard) {
    Tensor t1 = Tensor::construct1d({-1, 4, 0.5, 1, 2, 0, 12, 11, -2.5});
    Tensor t2 = Tensor::construct1d({7, -2, 2, 3, 7, 10, -2, -1, -2});
    Tensor result = Tensor(t1.shape[0]);

    hadamard(t1, t2, result);

    ASSERT_EQ_1D(result, {-7, -8, 1, 3, 14, 0, -24, -11, 5});
}

TEST(tensor_operations_host, multiply_constant) {
    Tensor t1 = Tensor::construct1d({2, 3, -4, -5, -6, 7, 8, 9, 0});
    Tensor result = Tensor(t1.shape[0]);

    multiply(t1, -2, result);

    ASSERT_EQ_1D(result, {-4, -6, 8, 10, 12, -14, -16, -18, 0});
}

TEST(tensor_operations_host, multiply_matrix_vector) {
    Tensor matrix = Tensor(10, 12, 14, 15);
    Tensor vector = Tensor(12);
    Tensor result = Tensor(10);

    matrix(3, 5, 13, 9) = 7;
    std::cout << matrix.host[3 * 12 * 14 * 15 + 5 * 14 * 15 + 13 * 15 + 9] << std::endl;

//    for (size_t i = 0; i < matrix.shape[0]; i++) {
//        for (size_t j = 0; j < matrix.shape[1]; j++) {
//            matrix.
//        }
//    }

//    hadamard(t1, t2, result);

//    ASSERT_EQ_1D(result, {-7, -8, 1, 3, 14, 0, -24, -11, 5});
}


