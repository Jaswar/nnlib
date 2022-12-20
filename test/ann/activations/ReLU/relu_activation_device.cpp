/**
 * @file relu_activation_host.cpp
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

TEST(relu_activation_device, forward) {
    Tensor tensor = Tensor::construct2d({{1, -2, 3}, {3, -5, 0}});
    Tensor result = Tensor(tensor.shape[0], tensor.shape[1]);

    tensor.move(DEVICE);
    result.move(DEVICE);

    ReLUActivation activation = ReLUActivation();

    activation.forward(tensor, result);

    result.move(HOST);

    ASSERT_TENSOR_EQ_2D(result, {{1, 0, 3}, {3, 0, 0}});
}

TEST(relu_activation_device, compute_derivatives) {
    Tensor tensor = Tensor::construct2d({{1, -2, 3}, {3, -5, 0}});
    Tensor result = Tensor(tensor.shape[0], tensor.shape[1]);

    tensor.move(DEVICE);
    result.move(DEVICE);

    ReLUActivation activation = ReLUActivation();

    activation.computeDerivatives(tensor, result);

    result.move(HOST);

    ASSERT_TENSOR_EQ_2D(result, {{1, 0, 1}, {1, 0, 0}});
}

#endif
