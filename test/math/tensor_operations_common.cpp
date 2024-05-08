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
#include <algorithm>
#include "tensor_operations_common.h"

void tensorSumPBT(bool testDevice) {
    const auto size = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e5));
    const auto data = rcFloatVectorInRange(size, -1, 1);

    sTensor t = std::make_shared<Tensor>(Tensor::construct1d(data));
    sTensor result = std::make_shared<Tensor>(1);
    sTensor expected = std::make_shared<Tensor>(1);

    float total = 0;
    for (size_t i = 0; i < size; i++) {
        total += t->data[i];
    }
    expected->data[0] = total;

    if (testDevice) {
        t->move(DEVICE);
        result->move(DEVICE);
    }

    result = no_grad::sum(t);

    if (testDevice) {
        result->move(HOST);
    }

    RC_ASSERT_TENSOR_CLOSE(*result, *expected, 1e-5, true);
}

void tensorFillPBT(float value, bool testDevice) {
    const auto size = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e6));

    sTensor result = std::make_shared<Tensor>(size);
    sTensor expected = std::make_shared<Tensor>(size);

    for (size_t i = 0; i < size; i++) {
        expected->data[i] = value;
    }

    if (testDevice) {
        result->move(DEVICE);
    }

    fill(value, result);

    if (testDevice) {
        result->move(HOST);
    }

    RC_ASSERT_TENSOR_EQ(*result, *expected);
}

void tensorAddPBT(bool testDevice) {
    const auto size = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e5));
    const auto data1 = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>()));
    const auto data2 = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>()));

    sTensor t1 = std::make_shared<Tensor>(Tensor::construct1d(data1));
    sTensor t2 = std::make_shared<Tensor>(Tensor::construct1d(data2));
    sTensor result = std::make_shared<Tensor>(size);
    sTensor expected = std::make_shared<Tensor>(size);

    for (size_t i = 0; i < size; i++) {
        expected->data[i] = t1->data[i] + t2->data[i];
    }

    if (testDevice) {
        t1->move(DEVICE);
        t2->move(DEVICE);
        result->move(DEVICE);
    }

    result = no_grad::add(t1, t2);

    if (testDevice) {
        result->move(HOST);
    }

    RC_ASSERT_TENSOR_EQ(*result, *expected);
}

void tensorAddBroadcastPBT(bool testDevice) {
    const auto n = *NO_SHRINK(rc::gen::inRange<size_t>(1, 2e3));
    const auto m = *NO_SHRINK(rc::gen::inRange<size_t>(1, 2e3));

    const auto dataMatrix = *NO_SHRINK(rc::gen::container<std::vector<float>>(n * m, rc::gen::arbitrary<float>()));
    const auto dataVector = *NO_SHRINK(rc::gen::container<std::vector<float>>(m, rc::gen::arbitrary<float>()));

    sTensor matrix = std::make_shared<Tensor>(n, m);
    std::copy(dataMatrix.begin(), dataMatrix.end(), matrix->data);
    sTensor vector = std::make_shared<Tensor>(Tensor::construct1d(dataVector));
    sTensor result = std::make_shared<Tensor>(n, m);
    sTensor expected = std::make_shared<Tensor>(n, m);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            expected->data[i * m + j] = matrix->data[i * m + j] + vector->data[j];
        }
    }

    if (testDevice) {
        matrix->move(DEVICE);
        vector->move(DEVICE);
        result->move(DEVICE);
    }

    result = no_grad::add(matrix, vector);

    if (testDevice) {
        result->move(HOST);
    }

    RC_ASSERT_TENSOR_EQ(*result, *expected);
}

void tensorSubtractPBT(bool testDevice) {
    const auto size = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e5));
    const auto data1 = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>()));
    const auto data2 = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>()));

    sTensor t1 = std::make_shared<Tensor>(Tensor::construct1d(data1));
    sTensor t2 = std::make_shared<Tensor>(Tensor::construct1d(data2));
    sTensor result = std::make_shared<Tensor>(size);
    sTensor expected = std::make_shared<Tensor>(size);

    for (size_t i = 0; i < size; i++) {
        expected->data[i] = t1->data[i] - t2->data[i];
    }

    if (testDevice) {
        t1->move(DEVICE);
        t2->move(DEVICE);
        result->move(DEVICE);
    }

    result = no_grad::subtract(t1, t2);

    if (testDevice) {
        result->move(HOST);
    }

    RC_ASSERT_TENSOR_EQ(*result, *expected);
}

void tensorHadamardPBT(bool testDevice) {
    const auto size = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e5));
    const auto data1 = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>()));
    const auto data2 = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>()));

    sTensor t1 = std::make_shared<Tensor>(Tensor::construct1d(data1));
    sTensor t2 = std::make_shared<Tensor>(Tensor::construct1d(data2));
    sTensor result = std::make_shared<Tensor>(size);
    sTensor expected = std::make_shared<Tensor>(size);

    for (size_t i = 0; i < size; i++) {
        expected->data[i] = t1->data[i] * t2->data[i];
    }

    if (testDevice) {
        t1->move(DEVICE);
        t2->move(DEVICE);
        result->move(DEVICE);
    }

    result = hadamard(t1, t2);

    if (testDevice) {
        result->move(HOST);
    }

    RC_ASSERT_TENSOR_EQ(*result, *expected);
}

void tensorDividePBT(bool testDevice) {
    const auto size = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e5));
    const auto data1 = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>()));
    const auto data2 = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::nonZero<float>()));

    sTensor t1 = std::make_shared<Tensor>(Tensor::construct1d(data1));
    sTensor t2 = std::make_shared<Tensor>(Tensor::construct1d(data2));
    sTensor result = std::make_shared<Tensor>(size);
    sTensor expected = std::make_shared<Tensor>(size);

    for (size_t i = 0; i < size; i++) {
        expected->data[i] = t1->data[i] / t2->data[i];
    }

    if (testDevice) {
        t1->move(DEVICE);
        t2->move(DEVICE);
        result->move(DEVICE);
    }

    result = no_grad::divide(t1, t2);

    if (testDevice) {
        result->move(HOST);
    }

    RC_ASSERT_TENSOR_EQ(*result, *expected);
}

void tensorLogPBT(bool testDevice) {
    const auto size = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e5));
    const auto data = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::positive<float>()));

    sTensor t = std::make_shared<Tensor>(Tensor::construct1d(data));
    sTensor result = std::make_shared<Tensor>(size);
    sTensor expected = std::make_shared<Tensor>(size);

    for (size_t i = 0; i < size; i++) {
        expected->data[i] = log(t->data[i]);
    }

    if (testDevice) {
        t->move(DEVICE);
        result->move(DEVICE);
    }

    result = no_grad::log(t);

    if (testDevice) {
        result->move(HOST);
    }

    // Need to use CLOSE here probably because of different implementations of log in CPU and GPU
    RC_ASSERT_TENSOR_CLOSE(*result, *expected);
}

void tensorMultiplyConstantPBT(float constant, bool testDevice) {
    const auto size = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e5));
    const auto data = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>()));

    sTensor t = std::make_shared<Tensor>(Tensor::construct1d(data));
    sTensor result = std::make_shared<Tensor>(size);
    sTensor expected = std::make_shared<Tensor>(size);

    for (size_t i = 0; i < size; i++) {
        expected->data[i] = t->data[i] * constant;
    }

    if (testDevice) {
        t->move(DEVICE);
        result->move(DEVICE);
    }

    result = no_grad::multiply(t, constant);

    if (testDevice) {
        result->move(HOST);
    }

    RC_ASSERT_TENSOR_CLOSE(*result, *expected);
}

void tensorMultiplyMatrixVectorPBT(bool testDevice) {
    const auto n = *NO_SHRINK(rc::gen::inRange<size_t>(1, 2e3));
    const auto m = *NO_SHRINK(rc::gen::inRange<size_t>(1, 2e3));

    const std::vector<float> dataMatrix = rcFloatVectorInRange(n * m, -1, 1);
    const std::vector<float> dataVector = rcFloatVectorInRange(m, -1, 1);

    sTensor matrix = std::make_shared<Tensor>(n, m);
    std::copy(dataMatrix.begin(), dataMatrix.end(), matrix->data);
    sTensor vector = std::make_shared<Tensor>(Tensor::construct1d(dataVector));
    sTensor result = std::make_shared<Tensor>(n);
    sTensor expected = std::make_shared<Tensor>(n);

    for (size_t i = 0; i < n; i++) {
        float acc = 0;
        for (size_t j = 0; j < m; j++) {
            acc += matrix->data[i * m + j] * vector->data[j];
        }
        expected->data[i] = acc;
    }

    if (testDevice) {
        matrix->move(DEVICE);
        vector->move(DEVICE);
        result->move(DEVICE);
    }

    result = no_grad::multiply(matrix, vector);

    if (testDevice) {
        result->move(HOST);
    }

    RC_ASSERT_TENSOR_CLOSE(*result, *expected, 5e-4);
}

void tensorMultiplyMatrixMatrixPBT(bool testDevice) {
    const auto n = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1500));
    const auto m = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1500));
    const auto k = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1500));

    std::vector<float> dataM1 = rcFloatVectorInRange(n * m, -1, 1);
    std::vector<float> dataM2 = rcFloatVectorInRange(m * k, -1, 1);

    sTensor m1 = std::make_shared<Tensor>(n, m);
    std::copy(dataM1.begin(), dataM1.end(), m1->data);
    sTensor m2 = std::make_shared<Tensor>(m, k);
    std::copy(dataM2.begin(), dataM2.end(), m2->data);
    sTensor result = std::make_shared<Tensor>(n, k);
    sTensor expected = std::make_shared<Tensor>(n, k);

    for (size_t row = 0; row < n; row++) {
        for (size_t column = 0; column < k; column++) {
            float acc = 0;
            for (size_t i = 0; i < m; i++) {
                acc += m1->data[row * m + i] * m2->data[i * k + column];
            }
            expected->data[row * k + column] = acc;
        }
    }

    if (testDevice) {
        m1->move(DEVICE);
        m2->move(DEVICE);
        result->move(DEVICE);
    }

    result = no_grad::multiply(m1, m2);

    if (testDevice) {
        result->move(HOST);
    }

    RC_ASSERT_TENSOR_CLOSE(*result, *expected, 5e-4);
}

void tensorTransposePBT(bool testDevice) {
    const auto n = *NO_SHRINK(rc::gen::inRange<size_t>(1, 2e3));
    const auto m = *NO_SHRINK(rc::gen::inRange<size_t>(1, 2e3));

    const auto data = *NO_SHRINK(rc::gen::container<std::vector<float>>(n * m, rc::gen::arbitrary<float>()));

    sTensor t = std::make_shared<Tensor>(n, m);
    std::copy(data.begin(), data.end(), t->data);
    sTensor result = std::make_shared<Tensor>(m, n);
    sTensor expected = std::make_shared<Tensor>(m, n);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            expected->data[j * n + i] = t->data[i * m + j];
        }
    }

    if (testDevice) {
        t->move(DEVICE);
        result->move(DEVICE);
    }

    result = no_grad::transpose(t);

    if (testDevice) {
        result->move(HOST);
    }

    RC_ASSERT_TENSOR_EQ(*result, *expected);
}

void tensorReluPBT(bool testDevice) {
    const auto size = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e5));
    const auto data = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::positive<float>()));

    sTensor t = std::make_shared<Tensor>(Tensor::construct1d(data));
    sTensor result = std::make_shared<Tensor>(size);
    sTensor expected = std::make_shared<Tensor>(size);

    for (size_t i = 0; i < size; i++) {
        if (t->data[i] > 0) {
            expected->data[i] = t->data[i];
        } else {
            expected->data[i] = 0;
        }
    }

    if (testDevice) {
        t->move(DEVICE);
        result->move(DEVICE);
    }

    result = no_grad::relu(t);

    if (testDevice) {
        result->move(HOST);
    }

    // Need to use CLOSE here probably because of different implementations of log in CPU and GPU
    RC_ASSERT_TENSOR_CLOSE(*result, *expected);
}

void tensorSigmoidPBT(bool testDevice) {
    const auto size = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e5));
    const auto data = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::positive<float>()));

    sTensor t = std::make_shared<Tensor>(Tensor::construct1d(data));
    sTensor result = std::make_shared<Tensor>(size);
    sTensor expected = std::make_shared<Tensor>(size);

    for (size_t i = 0; i < size; i++) {
        expected->data[i] = 1 / (1 + expf(-t->data[i]));
    }

    if (testDevice) {
        t->move(DEVICE);
        result->move(DEVICE);
    }

    result = no_grad::sigmoid(t);

    if (testDevice) {
        result->move(HOST);
    }

    // Need to use CLOSE here probably because of different implementations of log in CPU and GPU
    RC_ASSERT_TENSOR_CLOSE(*result, *expected);
}
