//
// Created by Jan Warchocki on 28/05/2022.
//


#include "../../../../include/activation.h"

ReLUActivation::ReLUActivation() : Activation(new ReLUOnHostEvaluator(), new ReLUOnDeviceEvaluator()) {
}
