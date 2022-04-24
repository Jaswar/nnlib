//
// Created by Jan Warchocki on 14/03/2022.
//

#ifndef NNLIB_VECTOR_OPERATIONS_CUH
#define NNLIB_VECTOR_OPERATIONS_CUH

#include "../../include/vector.h"

void addVectors(const Vector& v1, const Vector& v2, Vector& result);
void subtractVectors(const Vector& v1, const Vector& v2, Vector& result);
void multiplyVector(const Vector& v1, DTYPE constant, Vector& result);

#endif //NNLIB_VECTOR_OPERATIONS_CUH
