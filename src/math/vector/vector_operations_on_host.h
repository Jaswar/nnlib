//
// Created by Jan Warchocki on 29/05/2022.
//

#ifndef NNLIB_VECTOR_OPERATIONS_ON_HOST_H
#define NNLIB_VECTOR_OPERATIONS_ON_HOST_H

#include <vector.h>

void addVectorsOnHost(const Vector& v1, const Vector& v2, Vector& result);
void subtractVectorsOnHost(const Vector& v1, const Vector& v2, Vector& result);
void multiplyVectorOnHost(const Vector& v1, DTYPE constant, Vector& result);

#endif //NNLIB_VECTOR_OPERATIONS_ON_HOST_H
