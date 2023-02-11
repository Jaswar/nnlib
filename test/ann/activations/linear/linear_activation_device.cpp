/**
 * @file linear_activation_device.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 18 October 2022
 */

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <tensor.h>
#include "linear_activation_common.h"

#ifdef __CUDA__

RC_GTEST_PROP(linear_activation_device, forward, ()) {
    linearActivationForwardPBT(true);
}

RC_GTEST_PROP(linear_activation_device, compute_derivatives, ()) {
    linearActivationComputeDerivativesPBT(true);
}

#endif

