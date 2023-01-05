/**
 * @file tensor_operations_device.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 13 October 2022
 */

#include <verify.cuh>
#include <gtest/gtest.h>
#include <tensor.h>
#include "../assertions.h"
#include "../test_utils.h"
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <limits>

#ifdef __CUDA__

#define NO_SHRINK(...) rc::gen::noShrink(__VA_ARGS__)

RC_GTEST_PROP(tensor_operations_device, fill, (float value)) {
    const auto size = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e6));

    Tensor result = Tensor(size);
    Tensor expected = Tensor(size);

    for (size_t i = 0; i < size; i++) {
        expected.data[i] = value;
    }

    result.move(DEVICE);

    fill(value, result);

    result.move(HOST);

    RC_ASSERT_TENSOR_EQ(result, expected);
}

RC_GTEST_PROP(tensor_operations_device, add, ()) {
    const auto size = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e5));
    const auto data1 = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>()));
    const auto data2 = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>()));

    Tensor t1 = Tensor::construct1d(data1);
    Tensor t2 = Tensor::construct1d(data2);
    Tensor result = Tensor(size);
    Tensor expected = Tensor(size);

    for (size_t i = 0; i < size; i++) {
        expected.data[i] = t1.data[i] + t2.data[i];
    }

    t1.move(DEVICE);
    t2.move(DEVICE);
    result.move(DEVICE);

    add(t1, t2, result);

    result.move(HOST);

    RC_ASSERT_TENSOR_EQ(result, expected);
}

TEST(tensor_operations_device, add_broadcast) {
    Tensor t1 = Tensor::construct2d({{2, 3, 4}, {5, 6, 7}, {8, 9, 0}});
    Tensor t2 = Tensor::construct1d({5, 3, -5});
    Tensor result = Tensor(t1.shape[0], t1.shape[1]);

    t1.move(DEVICE);
    t2.move(DEVICE);
    result.move(DEVICE);

    add(t1, t2, result);

    result.move(HOST);

    ASSERT_TENSOR_EQ_2D(result, {{7, 6, -1}, {10, 9, 2}, {13, 12, -5}});
}

RC_GTEST_PROP(tensor_operations_device, subtract, ()) {
    const auto size = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e5));
    const auto data1 = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>()));
    const auto data2 = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>()));

    Tensor t1 = Tensor::construct1d(data1);
    Tensor t2 = Tensor::construct1d(data2);
    Tensor result = Tensor(size);
    Tensor expected = Tensor(size);

    for (size_t i = 0; i < size; i++) {
        expected.data[i] = t1.data[i] - t2.data[i];
    }

    t1.move(DEVICE);
    t2.move(DEVICE);
    result.move(DEVICE);

    subtract(t1, t2, result);

    result.move(HOST);

    RC_ASSERT_TENSOR_EQ(result, expected);
}

RC_GTEST_PROP(tensor_operations_device, hadamard, ()) {
    const auto size = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e5));
    const auto data1 = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>()));
    const auto data2 = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>()));

    Tensor t1 = Tensor::construct1d(data1);
    Tensor t2 = Tensor::construct1d(data2);
    Tensor result = Tensor(size);
    Tensor expected = Tensor(size);

    for (size_t i = 0; i < size; i++) {
        expected.data[i] = t1.data[i] * t2.data[i];
    }

    t1.move(DEVICE);
    t2.move(DEVICE);
    result.move(DEVICE);

    hadamard(t1, t2, result);

    result.move(HOST);

    RC_ASSERT_TENSOR_EQ(result, expected);
}

RC_GTEST_PROP(tensor_operations_device, divide, ()) {
    const auto size = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e5));
    const auto data1 = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>()));
    const auto data2 = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::nonZero<float>()));

    Tensor t1 = Tensor::construct1d(data1);
    Tensor t2 = Tensor::construct1d(data2);
    Tensor result = Tensor(size);
    Tensor expected = Tensor(size);

    for (size_t i = 0; i < size; i++) {
        expected.data[i] = t1.data[i] / t2.data[i];
    }

    t1.move(DEVICE);
    t2.move(DEVICE);
    result.move(DEVICE);

    divide(t1, t2, result);

    result.move(HOST);

    RC_ASSERT_TENSOR_EQ(result, expected);
}

RC_GTEST_PROP(tensor_operations_device, log, ()) {
    const auto size = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e5));
    const auto data = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::positive<float>()));

    Tensor t = Tensor::construct1d(data);
    Tensor result = Tensor(size);
    Tensor expected = Tensor(size);

    for (size_t i = 0; i < size; i++) {
        expected.data[i] = log(t.data[i]);
    }

    t.move(DEVICE);
    result.move(DEVICE);

    log(t, result);

    result.move(HOST);

    RC_ASSERT_TENSOR_CLOSE(result, expected);
}

TEST(tensor_operations_device, multiply_constant) {
    Tensor t1 = Tensor::construct1d({2, 3, -4, -5, -6, 7, 8, 9, 0});
    Tensor result = Tensor(t1.shape[0]);

    t1.move(DEVICE);
    result.move(DEVICE);

    multiply(t1, -2, result);

    result.move(HOST);

    ASSERT_TENSOR_EQ_1D(result, {-4, -6, 8, 10, 12, -14, -16, -18, 0});
}

TEST(tensor_operations_device, multiply_matrix_vector) {
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

    matrix.move(DEVICE);
    vector.move(DEVICE);
    result.move(DEVICE);

    multiply(matrix, vector, result);

    result.move(HOST);

    ASSERT_TENSOR_CLOSE(result, expected);
}

TEST(tensor_operations_device, multiply_matrix_matrix) {
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

    m1.move(DEVICE);
    m2.move(DEVICE);
    result.move(DEVICE);

    multiply(m1, m2, result);

    result.move(HOST);

    ASSERT_TENSOR_CLOSE(result, expected);
}

TEST(tensor_operations_device, transpose) {
    Tensor t1 = Tensor::construct2d({{1, 2, 3, 4}, {4, 5, 6, 7}, {7, 8, 9, 10}});
    Tensor result = Tensor(t1.shape[1], t1.shape[0]);

    t1.move(DEVICE);
    result.move(DEVICE);

    transpose(t1, result);

    result.move(HOST);

    ASSERT_TENSOR_EQ_2D(result, {{1, 4, 7}, {2, 5, 8}, {3, 6, 9}, {4, 7, 10}});
}

#endif
