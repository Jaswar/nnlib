/**
 * @file linear_activation.cpp
 * @brief Source file defining the LinearActivation class.
 * @author Jan Warchocki
 * @date 28 May 2022
 */

#include "../../../../include/activation.h"
#include <exceptions/different_data_location_exception.h>
#include <utils/location_verifiers.h>

LinearActivation::LinearActivation() : Activation(new LinearOnHostEvaluator(), new LinearOnDeviceEvaluator()) {
}
