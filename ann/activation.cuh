//
// Created by Jan Warchocki on 03/03/2022.
//

#ifndef NNLIB_ACTIVATION_CUH
#define NNLIB_ACTIVATION_CUH

#include "../math/vector.h"
#include "../gpu/verify.cuh"

void ReLU(const Vector& v, Vector& result);
void sigmoid(const Vector& v, Vector& result);
void linear(const Vector& v, Vector& result);

void ReLUDerivative(const Vector& input, Vector& result);
void sigmoidDerivative(const Vector& input, Vector& result);
void linearDerivative(const Vector& input, Vector& result);

#endif //NNLIB_ACTIVATION_CUH
