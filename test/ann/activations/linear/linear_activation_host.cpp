/**
 * @file linear_activation_host.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 14 October 2022
 */


#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <tensor.h>
#include "linear_activation_common.h"

RC_GTEST_PROP(linear_activation_host, forward, ()) {
    linearActivationForwardPBT(false);
}

RC_GTEST_PROP(linear_activation_host, compute_derivatives, ()) {
    linearActivationComputeDerivativesPBT(false);
}
