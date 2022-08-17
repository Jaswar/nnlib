/**
 * @file sigmoid_activation.cpp
 * @brief Source file defining the SigmoidActivation class.
 * @author Jan Warchocki
 * @date 28 May 2022
 */

#include "../../../../include/activation.h"

SigmoidActivation::SigmoidActivation() : Activation(new SigmoidOnHostEvaluator(), new SigmoidOnDeviceEvaluator()) {
}
