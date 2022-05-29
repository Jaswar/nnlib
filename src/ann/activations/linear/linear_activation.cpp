//
// Created by Jan Warchocki on 28/05/2022.
//

#include <utils/location_verifiers.h>
#include <exceptions/different_data_location_exception.h>
#include <activation.h>

LinearActivation::LinearActivation() : Activation(new LinearOnHostEvaluator(),
                                                  new LinearOnDeviceEvaluator()) {
}
