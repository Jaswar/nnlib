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

RC_GTEST_PROP(tensor_operations_device, add_broadcast, ()) {
    const auto n = *NO_SHRINK(rc::gen::inRange<size_t>(1, 2e3));
    const auto m = *NO_SHRINK(rc::gen::inRange<size_t>(1, 2e3));

    const auto dataMatrix = *NO_SHRINK(rc::gen::container<std::vector<float>>(n * m, rc::gen::arbitrary<float>()));
    const auto dataVector = *NO_SHRINK(rc::gen::container<std::vector<float>>(m, rc::gen::arbitrary<float>()));

    Tensor matrix = Tensor(n, m);
    std::copy(dataMatrix.begin(), dataMatrix.end(), matrix.data);
    Tensor vector = Tensor::construct1d(dataVector);
    Tensor result = Tensor(n, m);
    Tensor expected = Tensor(n, m);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            expected.data[i * m + j] = matrix.data[i * m + j] + vector.data[j];
        }
    }

    matrix.move(DEVICE);
    vector.move(DEVICE);
    result.move(DEVICE);

    add(matrix, vector, result);

    result.move(HOST);

    RC_ASSERT_TENSOR_EQ(result, expected);
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

    // Need to use CLOSE here probably because of different implementations of log in CPU and GPU
    RC_ASSERT_TENSOR_CLOSE(result, expected);
}

RC_GTEST_PROP(tensor_operations_device, multiply_constant, (float constant)) {
    const auto size = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e5));
    const auto data = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>()));

    Tensor t = Tensor::construct1d(data);
    Tensor result = Tensor(size);
    Tensor expected = Tensor(size);

    for (size_t i = 0; i < size; i++) {
        expected.data[i] = t.data[i] * constant;
    }

    t.move(DEVICE);
    result.move(DEVICE);

    multiply(t, constant, result);

    result.move(HOST);

    RC_ASSERT_TENSOR_CLOSE(result, expected);
}

RC_GTEST_PROP(tensor_operations_device, multiply_matrix_vector, ()) {
    const auto n = *NO_SHRINK(rc::gen::inRange<size_t>(1, 2e3));
    const auto m = *NO_SHRINK(rc::gen::inRange<size_t>(1, 2e3));

    const auto dataMatrixInt = *NO_SHRINK(rc::gen::container<std::vector<int>>(n * m, rc::gen::inRange<int>(-1e6, 1e6)));
    const auto dataVectorInt = *NO_SHRINK(rc::gen::container<std::vector<int>>(m, rc::gen::inRange<int>(-1e6, 1e6)));
    std::vector<float> dataMatrix = std::vector<float>(n * m);
    std::vector<float> dataVector = std::vector<float>(m);
    std::transform(dataMatrixInt.begin(), dataMatrixInt.end(), dataMatrix.begin(), [](int x) {
        return static_cast<float>(x);
    });
    std::transform(dataVectorInt.begin(), dataVectorInt.end(), dataVector.begin(), [](int x) {
        return static_cast<float>(x);
    });


    Tensor matrix = Tensor(n, m);
    std::copy(dataMatrix.begin(), dataMatrix.end(), matrix.data);
    Tensor vector = Tensor::construct1d(dataVector);

    multiply(matrix, 1e-6, matrix);
    multiply(vector, 1e-6, vector);

    Tensor result = Tensor(n);
    Tensor expected = Tensor(n);

    for (size_t i = 0; i < n; i++) {
        float acc = 0;
        for (size_t j = 0; j < m; j++) {
            acc += matrix.data[i * m + j] * vector.data[j];
        }
        expected.data[i] = acc;
    }

    matrix.move(DEVICE);
    vector.move(DEVICE);
    result.move(DEVICE);

    multiply(matrix, vector, result);

    result.move(HOST);

    RC_ASSERT_TENSOR_CLOSE(result, expected, 5e-4);
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

RC_GTEST_PROP(tensor_operations_device, transpose, ()) {
    const auto n = *NO_SHRINK(rc::gen::inRange<size_t>(1, 2e3));
    const auto m = *NO_SHRINK(rc::gen::inRange<size_t>(1, 2e3));

    const auto data = *NO_SHRINK(rc::gen::container<std::vector<float>>(n * m, rc::gen::arbitrary<float>()));

    Tensor t = Tensor(n, m);
    std::copy(data.begin(), data.end(), t.data);
    Tensor result = Tensor(m, n);
    Tensor expected = Tensor(m, n);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            expected.data[j * n + i] = t.data[i * m + j];
        }
    }

    t.move(DEVICE);
    result.move(DEVICE);

    transpose(t, result);

    result.move(HOST);

    RC_ASSERT_TENSOR_EQ(result, expected);
}

#endif
