//
// Created by Jan Warchocki on 15/03/2022.
//

#include "matrix.h"

#ifndef NNLIB_MATRIX_OPERATIONS_CUH
#define NNLIB_MATRIX_OPERATIONS_CUH

void addMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result);
void addBroadcastOnDevice(const Matrix& m, const Vector& v, Matrix& result);
void subtractMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result);

void multiplyMatrixVectorOnDevice(const Matrix& matrix, const Vector& vector, Vector& result);
void multiplyMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result);
void multiplyMatrixOnDevice(const Matrix& m1, DTYPE constant, Matrix& result);

void hadamardMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result);

void transposeMatrixOnDevice(const Matrix& m, Matrix& result);

#endif //NNLIB_MATRIX_OPERATIONS_CUH
