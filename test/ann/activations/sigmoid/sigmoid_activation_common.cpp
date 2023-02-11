/**
 * @file sigmoid_activation_common.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 11 February 2023
 */

#include "sigmoid_activation_common.h"
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <tensor.h>
#include <activation.h>
#include "../../../test_utils.h"
#include "../../../assertions.h"
#include <cmath>

float sigmoid(float x) {
    return 1 / (1 + expf(-x));
}

void sigmoidActivationForwardPBT(bool testDevice) {
    const auto size = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e5));
    const auto data1 = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>()));

    Tensor tensor = Tensor::construct1d(data1);
    Tensor result = Tensor(size);
    Tensor expected = Tensor(size);

    for (size_t i = 0; i < size; i++) {
        expected.data[i] = sigmoid(data1[i]);
    }

    if (testDevice) {
        tensor.move(DEVICE);
        result.move(DEVICE);
    }

    SigmoidActivation activation = SigmoidActivation();

    activation.forward(tensor, result);

    if (testDevice) {
        result.move(HOST);
    }

    RC_ASSERT_TENSOR_CLOSE(result, expected);
}

void sigmoidActivationComputeDerivativesPBT(bool testDevice) {
    const auto size = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e5));
    const auto data1 = *NO_SHRINK(rc::gen::container<std::vector<float>>(size, rc::gen::arbitrary<float>()));

    Tensor tensor = Tensor::construct1d(data1);
    Tensor result = Tensor(size);
    Tensor expected = Tensor(size);

    for (size_t i = 0; i < size; i++) {
        expected.data[i] = sigmoid(data1[i]) * (1 - sigmoid(data1[i]));
    }

    if (testDevice) {
        tensor.move(DEVICE);
        result.move(DEVICE);
    }

    SigmoidActivation activation = SigmoidActivation();

    activation.computeDerivatives(tensor, result);

    if (testDevice) {
        result.move(HOST);
    }

    RC_ASSERT_TENSOR_CLOSE(result, expected);
}
