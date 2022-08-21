/**
 * @file matrix_operations_on_host.h
 * @brief Header file declaring matrix operations that happen on host.
 *
 * The methods declared in this file are only called when all operands are located on the host.
 *
 * These methods do not perform any checking with regards to the size or location of the operands. This is already
 * done in the corresponding methods in matrix.cpp.
 *
 * @author Jan Warchocki
 * @date 29 May 2022
 */

#ifndef NNLIB_MATRIX_OPERATIONS_ON_HOST_H
#define NNLIB_MATRIX_OPERATIONS_ON_HOST_H

#include <matrix.h>

/** @copydoc matrix_operations_on_device.cuh::addMatricesOnDevice */
void addMatricesOnHost(const Matrix& m1, const Matrix& m2, Matrix& result);

/** @copydoc matrix_operations_on_device.cuh::addBroadcastOnDevice */
void addBroadcastOnHost(const Matrix& m, const Vector& v, Matrix& result);

/** @copydoc matrix_operations_on_device.cuh::subtractMatricesOnDevice */
void subtractMatricesOnHost(const Matrix& m1, const Matrix& m2, Matrix& result);

/** @copydoc matrix_operations_on_device.cuh::multiplyMatrixVectorOnDevice */
void multiplyMatrixVectorOnHost(const Matrix& matrix, const Vector& vector, Vector& result);

/** @copydoc matrix_operations_on_device.cuh::multiplyMatricesOnDevice */
void multiplyMatricesOnHost(const Matrix& m1, const Matrix& m2, Matrix& result);

/** @copydoc matrix_operations_on_device.cuh::multiplyMatrixOnDevice */
void multiplyMatrixOnHost(const Matrix& m, DTYPE constant, Matrix& result);

/** @copydoc matrix_operations_on_device.cuh::hadamardMatricesOnDevice */
void hadamardMatricesOnHost(const Matrix& m1, const Matrix& m2, Matrix& result);

/** @copydoc matrix_operations_on_device.cuh::transposeMatrixOnDevice */
void transposeMatrixOnHost(const Matrix& m, Matrix& result);

#endif //NNLIB_MATRIX_OPERATIONS_ON_HOST_H
