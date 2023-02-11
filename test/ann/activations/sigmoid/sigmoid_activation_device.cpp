/**
 * @file sigmoid_activation_device.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 18 October 2022
 */

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <tensor.h>
#include "sigmoid_activation_common.h"

#ifdef __CUDA__

RC_GTEST_PROP(sigmoid_activation_device, forward, ()) {
    sigmoidActivationForwardPBT(true);
}

RC_GTEST_PROP(sigmoid_activation_device, compute_derivatives, ()) {
    sigmoidActivationComputeDerivativesPBT(true);
}

#endif
