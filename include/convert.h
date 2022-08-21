/**
 * @file convert.h
 * @brief Header file consisting of function declarations to convert between matrices and vectors.
 * @author Jan Warchocki
 * @date 07 March 2022
 */

#ifndef NNLIB_CONVERT_H
#define NNLIB_CONVERT_H

#include "matrix.h"
#include "vector.h"

/**
 * @brief Convert a matrix to a vector.
 *
 * The matrix should be an n x 1 matrix (consisting of only one column).
 *
 * @param matrix The matrix to convert to a vector.
 * @return %Vector corresponding to the passed matrix.
 *
 * @throws SizeMismatchException If matrix has more than one column.
 */
Vector convertToVector(const Matrix& matrix);

/**
 * @brief Convert a vector to a matrix.
 *
 * The matrix returned as a result has one column and is therefore of the shape n x 1, where n is the size of
 * the vector.
 *
 * @param vector The vector to convert to a matrix.
 * @return The matrix corresponding to the passed vector.
 */
Matrix convertToMatrix(const Vector& vector);

#endif //NNLIB_CONVERT_H
