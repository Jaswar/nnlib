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

RC_GTEST_PROP(relu_activation_host, forward, ()) {
    reluActivationForwardPBT(false);
}

RC_GTEST_PROP(relu_activation_host, compute_derivatives, ()) {
    reluActivationComputeDerivativesPBT(false);
}

