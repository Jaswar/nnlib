/**
 * @file tensor_operations_common.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 25 January 2023
 */

#include <verify.cuh>
#include <gtest/gtest.h>
#include <tensor.h>
#include "../assertions.h"
#include "../test_utils.h"
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <limits>
#include "tensor_operations_common.h"

void tensorFillPBT(float value, bool testDevice) {
    const auto size = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e6));

    Tensor result = Tensor(size);
    Tensor expected = Tensor(size);

    for (size_t i = 0; i < size; i++) {
        expected.data[i] = value;
    }

    if (testDevice) {
        result.move(DEVICE);
    }

    fill(value, result);

    if (testDevice) {
        result.move(HOST);
    }

    RC_ASSERT_TENSOR_EQ(result, expected);
}

void tensorAddPBT(bool testDevice) {
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

    if (testDevice) {
        t1.move(DEVICE);
        t2.move(DEVICE);
        result.move(DEVICE);
    }

    add(t1, t2, result);

    if (testDevice) {
        result.move(HOST);
    }

    RC_ASSERT_TENSOR_EQ(result, expected);
}

void tensorAddBroadcastPBT(bool testDevice) {
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

    if (testDevice) {
        matrix.move(DEVICE);
        vector.move(DEVICE);
        result.move(DEVICE);
    }

    add(matrix, vector, result);

    if (testDevice) {
        result.move(HOST);
    }

    RC_ASSERT_TENSOR_EQ(result, expected);
}

void tensorSubtractPBT(bool testDevice) {
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

    if (testDevice) {
        t1.move(DEVICE);
        t2.move(DEVICE);
        result.move(DEVICE);
    }

    subtract(t1, t2, result);

    if (testDevice) {
        result.move(HOST);
    }

    RC_ASSERT_TENSOR_EQ(result, expected);
}

void tensorHadamardPBT(bool testDevice) {
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

    if (testDevice) {
        t1.move(DEVICE);
        t2.move(DEVICE);
        result.move(DEVICE);
    }

    hadamard(t1, t2, result);

    if (testDevice) {
        result.move(HOST);
    }

    RC_ASSERT_TENSOR_EQ(result, expected);
}

void tensorDividePBT(bool testDevice) {
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

    if (testDevice) {
        t1.move(DEVICE);
        t2.move(DEVICE);
        result.move(DEVICE);
    }

    divide(t1, t2, result);

    if (testDevice) {
        result.move(HOST);
    }

    RC_ASSERT_TENSOR_EQ(result, expected);
}

void tensorLogPBT(bool testDevice) {
    const auto size = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e5));
    const auto data = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::positive<float>()));

    Tensor t = Tensor::construct1d(data);
    Tensor result = Tensor(size);
    Tensor expected = Tensor(size);

    for (size_t i = 0; i < size; i++) {
        expected.data[i] = log(t.data[i]);
    }

    if (testDevice) {
        t.move(DEVICE);
        result.move(DEVICE);
    }

    log(t, result);

    if (testDevice) {
        result.move(HOST);
    }

    // Need to use CLOSE here probably because of different implementations of log in CPU and GPU
    RC_ASSERT_TENSOR_CLOSE(result, expected);
}

void tensorMultiplyConstantPBT(float constant, bool testDevice) {
    const auto size = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e5));
    const auto data = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>()));

    Tensor t = Tensor::construct1d(data);
    Tensor result = Tensor(size);
    Tensor expected = Tensor(size);

    for (size_t i = 0; i < size; i++) {
        expected.data[i] = t.data[i] * constant;
    }

    if (testDevice) {
        t.move(DEVICE);
        result.move(DEVICE);
    }

    multiply(t, constant, result);

    if (testDevice) {
        result.move(HOST);
    }

    RC_ASSERT_TENSOR_CLOSE(result, expected);
}

void tensorMultiplyMatrixVectorPBT(bool testDevice) {
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

    if (testDevice) {
        matrix.move(DEVICE);
        vector.move(DEVICE);
        result.move(DEVICE);
    }

    multiply(matrix, vector, result);

    if (testDevice) {
        result.move(HOST);
    }

    RC_ASSERT_TENSOR_CLOSE(result, expected, 5e-4);
}

void tensorMultiplyMatrixMatrixPBT(bool testDevice) {
    const auto n = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1500));
    const auto m = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1500));
    const auto k = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1500));

    const auto dataM1Int = *NO_SHRINK(rc::gen::container<std::vector<int>>(n * m, rc::gen::inRange<int>(-1e6, 1e6)));
    const auto dataM2Int = *NO_SHRINK(rc::gen::container<std::vector<int>>(m * k, rc::gen::inRange<int>(-1e6, 1e6)));
    std::vector<float> dataM1 = std::vector<float>(n * m);
    std::vector<float> dataM2 = std::vector<float>(m * k);
    std::transform(dataM1Int.begin(), dataM1Int.end(), dataM1.begin(), [](int x) {
        return static_cast<float>(x);
    });
    std::transform(dataM2Int.begin(), dataM2Int.end(), dataM2.begin(), [](int x) {
        return static_cast<float>(x);
    });


    Tensor m1 = Tensor(n, m);
    std::copy(dataM1.begin(), dataM1.end(), m1.data);
    Tensor m2 = Tensor(m, k);
    std::copy(dataM2.begin(), dataM2.end(), m2.data);

    multiply(m1, 1e-6, m1);
    multiply(m2, 1e-6, m2);

    Tensor result = Tensor(n, k);
    Tensor expected = Tensor(n, k);

    for (size_t row = 0; row < n; row++) {
        for (size_t column = 0; column < k; column++) {
            float acc = 0;
            for (size_t i = 0; i < m; i++) {
                acc += m1.data[row * m + i] * m2.data[i * k + column];
            }
            expected.data[row * k + column] = acc;
        }
    }

    if (testDevice) {
        m1.move(DEVICE);
        m2.move(DEVICE);
        result.move(DEVICE);
    }

    multiply(m1, m2, result);

    if (testDevice) {
        result.move(HOST);
    }

    RC_ASSERT_TENSOR_CLOSE(result, expected, 5e-4);
}

void tensorTransposePBT(bool testDevice) {
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

    if (testDevice) {
        t.move(DEVICE);
        result.move(DEVICE);
    }

    transpose(t, result);

    if (testDevice) {
        result.move(HOST);
    }

    RC_ASSERT_TENSOR_EQ(result, expected);
}
