//
// Created by Jan Warchocki on 03/03/2022.
//

#ifndef NNLIB_ACTIVATION_CUH
#define NNLIB_ACTIVATION_CUH

#include "../math/vector.h"

Vector ReLU(const Vector& v);
Vector sigmoid(const Vector& v);
Vector tanh(const Vector& v);

Vector ReLUDerivative(const Vector& input);
Vector sigmoidDerivative(const Vector& input);
Vector tanhDerivative(const Vector& input);
Vector linearDerivative(const Vector& input);

#endif //NNLIB_ACTIVATION_CUH
