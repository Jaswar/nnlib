//
// Created by Jan Warchocki on 07/03/2022.
//

#ifndef NNLIB_CONVERT_H
#define NNLIB_CONVERT_H

#include "matrix.h"
#include "vector.h"

Vector convertToVector(const Matrix& matrix);

Matrix convertToMatrix(const Vector& vector);

#endif //NNLIB_CONVERT_H
