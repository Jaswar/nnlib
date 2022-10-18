/**
 * @file linear_activation_host.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 14 October 2022
 */


#include <gtest/gtest.h>
#include <tensor.h>
#include <activation.h>
#include "../../../assertions.h"

TEST(linear_activation_host, forward) {
    Tensor tensor = Tensor::construct1d({2, 3, -3, 5, 1, 2});
    Tensor result = Tensor(tensor.shape[0]);

    LinearActivation activation = LinearActivation();

    activation.forward(tensor, result);

    ASSERT_TENSOR_EQ_1D(result, {2, 3, -3, 5, 1, 2});
}

TEST(linear_activation_host, compute_derivatives) {
    Tensor tensor = Tensor::construct1d({2, 3, -3, 5, 1, 2});
    Tensor result = Tensor(tensor.shape[0]);

    LinearActivation activation = LinearActivation();

    activation.computeDerivatives(tensor, result);

    ASSERT_TENSOR_EQ_1D(result, {1, 1, 1, 1, 1, 1});
}
