/**
 * @file vector_operations_on_host.h
 * @brief Header file declaring vector operations that happen on host.
 *
 * The methods declared in this file are only called when all operands are located on the host.
 *
 * These methods do not perform any checking with regards to the size or location of the operands. This is already
 * done in the corresponding methods in vector.cpp.
 *
 * @author Jan Warchocki
 * @date 29 May 2022
 */

#ifndef NNLIB_VECTOR_OPERATIONS_ON_HOST_H
#define NNLIB_VECTOR_OPERATIONS_ON_HOST_H

#include <vector.h>

/** @copydoc vector_operations_on_device.cuh::addVectorsOnDevice */
void addVectorsOnHost(const Vector& v1, const Vector& v2, Vector& result);

/** @copydoc vector_operations_on_device.cuh::subtractVectorsOnDevice */
void subtractVectorsOnHost(const Vector& v1, const Vector& v2, Vector& result);

/** @copydoc vector_operations_on_device.cuh::multiplyVectorOnDevice */
void multiplyVectorOnHost(const Vector& v1, DTYPE constant, Vector& result);

#endif //NNLIB_VECTOR_OPERATIONS_ON_HOST_H
