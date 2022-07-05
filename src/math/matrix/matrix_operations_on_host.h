//
// Created by Jan Warchocki on 29/05/2022.
//

#ifndef NNLIB_MATRIX_OPERATIONS_ON_HOST_H
#define NNLIB_MATRIX_OPERATIONS_ON_HOST_H

#include <matrix.h>

void addMatricesOnHost(const Matrix& m1, const Matrix& m2, Matrix& result);
void addBroadcastOnHost(const Matrix& m, const Vector& v, Matrix& result);
void subtractMatricesOnHost(const Matrix& m1, const Matrix& m2, Matrix& result);

void multiplyMatrixVectorOnHost(const Matrix& m, const Vector& v, Vector& result);
void multiplyMatricesOnHost(const Matrix& m1, const Matrix& m2, Matrix& result);
void multiplyMatrixOnHost(const Matrix& m, DTYPE constant, Matrix& result);

void hadamardMatricesOnHost(const Matrix& m1, const Matrix& m2, Matrix& result);

void transposeMatrixOnHost(const Matrix& m, Matrix& result);

#endif //NNLIB_MATRIX_OPERATIONS_ON_HOST_H
