//
// Created by Jan Warchocki on 14/03/2022.
//

#ifndef NNLIB_VECTOR_OPERATIONS_CUH
#define NNLIB_VECTOR_OPERATIONS_CUH

#include "../math/vector.h"

Vector addVectors(const Vector& v1, const Vector& v2);
Vector subtractVectors(const Vector& v1, const Vector& v2);
Vector multiplyVector(const Vector& v1, DTYPE constant);
Vector dotProduct(const Vector& v1, const Vector& v2);

#endif //NNLIB_VECTOR_OPERATIONS_CUH
