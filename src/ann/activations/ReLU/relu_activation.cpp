/**
 * @file relu_activation.cpp
 * @brief Source file defining the ReLUActivation class.
 * @author Jan Warchocki
 * @date 28 May 2022
 */


#include "../../../../include/activation.h"

ReLUActivation::ReLUActivation() : Activation(new ReLUOnHostEvaluator(), new ReLUOnDeviceEvaluator()) {
}
