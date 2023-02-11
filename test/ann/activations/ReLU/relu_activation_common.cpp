/**
 * @file relu_activation_common.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 11 February 2023
 */

#include "relu_activation_common.h"
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <tensor.h>
#include <activation.h>
#include "../../../test_utils.h"
#include "../../../assertions.h"

void reluActivationForwardPBT(bool testDevice) {
    const auto size = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e5));
    const auto data1 = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>()));

    Tensor tensor = Tensor::construct1d(data1);
    Tensor result = Tensor(size);
    Tensor expected = Tensor(size);

    for (size_t i = 0; i < size; i++) {
        if (data1[i] > 0) {
            expected.data[i] = data1[i];
        } else {
            expected.data[i] = 0;
        }
    }

    if (testDevice) {
        tensor.move(DEVICE);
        result.move(DEVICE);
    }

    ReLUActivation activation = ReLUActivation();

    activation.forward(tensor, result);

    if (testDevice) {
        result.move(HOST);
    }

    RC_ASSERT_TENSOR_EQ(result, expected);
}

void reluActivationComputeDerivativesPBT(bool testDevice) {
    const auto size = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e5));
    const auto data1 = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>()));

    Tensor tensor = Tensor::construct1d(data1);
    Tensor result = Tensor(size);
    Tensor expected = Tensor(size);

    for (size_t i = 0; i < size; i++) {
        if (data1[i] > 0) {
            expected.data[i] = 1;
        } else {
            expected.data[i] = 0;
        }
    }

    if (testDevice) {
        tensor.move(DEVICE);
        result.move(DEVICE);
    }

    ReLUActivation activation = ReLUActivation();

    activation.computeDerivatives(tensor, result);

    if (testDevice) {
        result.move(HOST);
    }

    RC_ASSERT_TENSOR_EQ(result, expected);
}
