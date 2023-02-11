/**
 * @file relu_activation_host.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 18 October 2022
 */

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <tensor.h>
#include "relu_activation_common.h"

#ifdef __CUDA__

RC_GTEST_PROP(relu_activation_device, forward, ()) {
    reluActivationForwardPBT(true);
}

RC_GTEST_PROP(relu_activation_device, compute_derivatives, ()) {
    reluActivationComputeDerivativesPBT(true);
}

#endif
