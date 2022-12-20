/**
 * @file linear_activation_device.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 18 October 2022
 */

#include <gtest/gtest.h>
#include <tensor.h>
#include <activation.h>
#include "../../../assertions.h"
#include <verify.cuh>

#ifdef __CUDA__

TEST(linear_activation_device, forward) {
    Tensor tensor = Tensor::construct1d({2, 3, -3, 5, 1, 2});
    Tensor result = Tensor(tensor.shape[0]);

    tensor.move(DEVICE);
    result.move(DEVICE);

    LinearActivation activation = LinearActivation();

    activation.forward(tensor, result);

    result.move(HOST);

    ASSERT_TENSOR_EQ_1D(result, {2, 3, -3, 5, 1, 2});
}

TEST(linear_activation_device, compute_derivatives) {
    Tensor tensor = Tensor::construct1d({2, 3, -3, 5, 1, 2});
    Tensor result = Tensor(tensor.shape[0]);

    tensor.move(DEVICE);
    result.move(DEVICE);

    LinearActivation activation = LinearActivation();

    activation.computeDerivatives(tensor, result);

    result.move(HOST);

    ASSERT_TENSOR_EQ_1D(result, {1, 1, 1, 1, 1, 1});
}

#endif

