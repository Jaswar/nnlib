/**
 * @file sigmoid_activation_device.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 18 October 2022
 */

#include <gtest/gtest.h>
#include <tensor.h>
#include <activation.h>
#include "../../../assertions.h"
#include <verify.cuh>

#ifdef HAS_CUDA

TEST(sigmoid_activation_device, forward) {
    Tensor tensor = Tensor::construct2d({{0, -1, 1}, {2, 3, -5}});
    Tensor result = Tensor(tensor.shape[0], tensor.shape[1]);

    tensor.move(DEVICE);
    result.move(DEVICE);

    SigmoidActivation activation = SigmoidActivation();

    activation.forward(tensor, result);

    result.move(HOST);

    ASSERT_TENSOR_CLOSE_2D(result, {{0.5, 0.2689, 0.7310}, {0.8807, 0.9525, 0.0066}}, 1e-4);
}

TEST(sigmoid_activation_device, compute_derivatives) {
    Tensor tensor = Tensor::construct2d({{0, -1, 1}, {2, 3, -5}});
    Tensor result = Tensor(tensor.shape[0], tensor.shape[1]);

    tensor.move(DEVICE);
    result.move(DEVICE);

    SigmoidActivation activation = SigmoidActivation();

    activation.computeDerivatives(tensor, result);

    result.move(HOST);

    ASSERT_TENSOR_CLOSE_2D(result, {{0.25, 0.1966, 0.1966}, {0.1049, 0.0451, 0.0066}}, 1e-4);
}

#endif
