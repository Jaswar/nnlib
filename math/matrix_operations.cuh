//
// Created by Jan Warchocki on 15/03/2022.
//

#include "matrix.h"

#ifndef NNLIB_MATRIX_OPERATIONS_CUH
#define NNLIB_MATRIX_OPERATIONS_CUH

void addMatrices(const Matrix& m1, const Matrix& m2, Matrix& result);
void subtractMatrices(const Matrix& m1, const Matrix& m2, Matrix& result);
void multiplyMatrixVector(const Matrix& matrix, const Vector& vector, Vector& result);
void multiplyMatrix(const Matrix& m1, DTYPE constant, Matrix& result);
void transposeMatrix(const Matrix& m, Matrix& result);

#endif //NNLIB_MATRIX_OPERATIONS_CUH
