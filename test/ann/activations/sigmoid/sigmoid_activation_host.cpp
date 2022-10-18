/**
 * @file sigmoid_activation_host.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 18 October 2022
 */

#include <gtest/gtest.h>
#include <tensor.h>
#include <activation.h>
#include "../../../assertions.h"

TEST(sigmoid_activation_host, forward) {
    Tensor tensor = Tensor::construct2d({{0, -1, 1}, {2, 3, -5}});
    Tensor result = Tensor(tensor.shape[0], tensor.shape[1]);

    SigmoidActivation activation = SigmoidActivation();

    activation.forward(tensor, result);

    ASSERT_TENSOR_CLOSE_2D(result, {{0.5, 0.2689, 0.7310}, {0.8807, 0.9525, 0.0066}}, 1e-4);
}

TEST(sigmoid_activation_host, compute_derivatives) {
    Tensor tensor = Tensor::construct2d({{0, -1, 1}, {2, 3, -5}});
    Tensor result = Tensor(tensor.shape[0], tensor.shape[1]);

    SigmoidActivation activation = SigmoidActivation();

    activation.computeDerivatives(tensor, result);

    ASSERT_TENSOR_CLOSE_2D(result, {{0.25, 0.1966, 0.1966}, {0.1049, 0.0451, 0.0066}}, 1e-4);
}
