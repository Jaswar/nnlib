//
// Created by Jan Warchocki on 29/05/2022.
//

#include <activation.h>

SigmoidActivation::SigmoidActivation(DataLocation location) : Activation(location) {
    if (this->location == HOST) {
        this->evaluator = new SigmoidOnHostEvaluator();
    } else {
        this->evaluator = new SigmoidOnDeviceEvaluator();
    }
}
