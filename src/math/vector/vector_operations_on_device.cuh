/**
 * @file vector_operations_on_device.cuh
 * @brief Header file declaring vector operations that happen on device.
 *
 * The methods declared in this file are only called when all operands are located on the GPU.
 *
 * These methods do not perform any checking with regards to the size or location of the operands. This is already
 * done in the corresponding methods in vector.cpp.
 *
 * @author Jan Warchocki
 * @date 14 March 2022
 */

#ifndef NNLIB_VECTOR_OPERATIONS_ON_DEVICE_CUH
#define NNLIB_VECTOR_OPERATIONS_ON_DEVICE_CUH

#include "vector.h"

/**
 * @brief Add two vectors.
 *
 * @param v1 The first vector.
 * @param v2 The second vector.
 * @param result Where the result of the addition should be stored.
 */
void addVectorsOnDevice(const Vector& v1, const Vector& v2, Vector& result);

/**
 * @brief Subtract one vector from another.
 *
 * @param v1 The vector to subtract from.
 * @param v2 The vector that should be subtracted.
 * @param result Where the result of the subtraction should be stored.
 */
void subtractVectorsOnDevice(const Vector& v1, const Vector& v2, Vector& result);

/**
 * @brief Multiply a vector with a constant.
 *
 * @param v1 The vector to multiply.
 * @param constant The constant to multiply with.
 * @param result Where the result of the multiplication should be stored.
 */
void multiplyVectorOnDevice(const Vector& v1, DTYPE constant, Vector& result);

#endif //NNLIB_VECTOR_OPERATIONS_ON_DEVICE_CUH
