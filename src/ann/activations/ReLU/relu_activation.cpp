//
// Created by Jan Warchocki on 28/05/2022.
//


#include <activation.h>

ReLUActivation::ReLUActivation(DataLocation location) : Activation(location) {
    if (this->location == HOST) {
        this->evaluator = new ReLUOnHostEvaluator();
    } else {
        this->evaluator = new ReLUOnDeviceEvaluator();
    }
}
