//
// Created by Jan Warchocki on 29/05/2022.
//

#include <activation.h>

SigmoidActivation::SigmoidActivation() : Activation(new SigmoidOnHostEvaluator(),
                                                    new SigmoidOnDeviceEvaluator()) {
}
