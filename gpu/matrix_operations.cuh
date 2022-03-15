//
// Created by Jan Warchocki on 15/03/2022.
//

#include "../math/matrix.h"

#ifndef NNLIB_MATRIX_OPERATIONS_CUH
#define NNLIB_MATRIX_OPERATIONS_CUH

Matrix addMatrices(const Matrix& m1, const Matrix& m2);
Matrix subtractMatrices(const Matrix& m1, const Matrix& m2);
Vector multiplyMatrixVector(const Matrix& matrix, const Vector& vector);
Matrix multiplyMatrix(const Matrix& m1, DTYPE constant);

#endif //NNLIB_MATRIX_OPERATIONS_CUH
