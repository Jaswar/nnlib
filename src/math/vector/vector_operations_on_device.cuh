//
// Created by Jan Warchocki on 14/03/2022.
//

#ifndef NNLIB_VECTOR_OPERATIONS_ON_DEVICE_CUH
#define NNLIB_VECTOR_OPERATIONS_ON_DEVICE_CUH

#include "vector.h"

void addVectorsOnDevice(const Vector& v1, const Vector& v2, Vector& result);
void subtractVectorsOnDevice(const Vector& v1, const Vector& v2, Vector& result);
void multiplyVectorOnDevice(const Vector& v1, DTYPE constant, Vector& result);

#endif //NNLIB_VECTOR_OPERATIONS_ON_DEVICE_CUH
