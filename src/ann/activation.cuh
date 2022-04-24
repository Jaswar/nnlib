//
// Created by Jan Warchocki on 03/03/2022.
//

#ifndef NNLIB_ACTIVATION_CUH
#define NNLIB_ACTIVATION_CUH

#include "../../include/vector.h"
#include "../gpu/verify.cuh"
#include "../../include/matrix.h"

void linear(const Vector& v, Vector& result);
void ReLU(const Vector& v, Vector& result);
void sigmoid(const Vector& v, Vector& result);

void linear(const Matrix& m, Matrix& result);
void ReLU(const Matrix& m, Matrix& result);
void sigmoid(const Matrix& m, Matrix& result);

void linearDerivative(const Vector& input, Vector& result);
void ReLUDerivative(const Vector& input, Vector& result);
void sigmoidDerivative(const Vector& input, Vector& result);

void linearDerivative(const Matrix& input, Matrix& result);
void ReLUDerivative(const Matrix& input, Matrix& result);
void sigmoidDerivative(const Matrix& input, Matrix& result);

#endif //NNLIB_ACTIVATION_CUH
