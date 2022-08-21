/**
 * @file matrix_operations_on_device.cuh
 * @brief Header file declaring matrix operations that happen on device.
 *
 * The methods declared in this file are only called when all operands are located on the GPU.
 *
 * These methods do not perform any checking with regards to the size or location of the operands. This is already
 * done in the corresponding methods in matrix.cpp.
 *
 * @author Jan Warchocki
 * @date 15 March 2022
 */

#include "matrix.h"

#ifndef NNLIB_MATRIX_OPERATIONS_CUH
#define NNLIB_MATRIX_OPERATIONS_CUH

/**
 * @brief Add two matrices.
 *
 * @param m1 The first matrix.
 * @param m2 The second matrix.
 * @param result Where the result of the addition should be stored.
 */
void addMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result);

/**
 * @brief Broadcast a vector over the rows of the matrix and add it to the matrix.
 *
 * @param m The matrix.
 * @param v The vector to broadcast.
 * @param result Where the result of the broadcast operation should be stored.
 */
void addBroadcastOnDevice(const Matrix& m, const Vector& v, Matrix& result);

/**
 * @brief Subtract one matrix from another.
 *
 * @param m1 The matrix to subtract from.
 * @param m2 The matrix that should be subtracted.
 * @param result Where the result of the subtraction should be stored.
 */
void subtractMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result);

/**
 * @brief Multiply a matrix with a vector.
 *
 * @param matrix The matrix to multiply.
 * @param vector The vector to multiply.
 * @param result Where the result of the multiplication should be stored.
 */
void multiplyMatrixVectorOnDevice(const Matrix& matrix, const Vector& vector, Vector& result);

/**
 * @brief Multiply two matrices together.
 *
 * @param m1 The first matrix.
 * @param m2 The second matrix.
 * @param result Where the result of the multiplication should be written to.
 */
void multiplyMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result);

/**
 * @brief Multiply a matrix with a constant number.
 *
 * @param m The matrix to multiply.
 * @param constant The constant to multiply @p m1 with.
 * @param result Where the result of the multiplication should be stored.
 */
void multiplyMatrixOnDevice(const Matrix& m, DTYPE constant, Matrix& result);

/**
 * @brief Perform hadamard product on matrices.
 *
 * @param m1 The first matrix.
 * @param m2 The second matrix.
 * @param result Where the result of the element-wise multiplication should be stored.
 */
void hadamardMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result);

/**
 * @brief Transpose a matrix.
 *
 * @param m The matrix to transpose.
 * @param result Where the result of transposing the matrix should be stored.
 */
void transposeMatrixOnDevice(const Matrix& m, Matrix& result);

#endif //NNLIB_MATRIX_OPERATIONS_CUH
