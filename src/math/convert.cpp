/**
 * @file convert.cpp
 * @brief Source file consisting of function definitions to convert between matrices and vectors.
 * @author Jan Warchocki
 * @date 07 March 2022
 */

#include "convert.h"
#include <exceptions/size_mismatch_exception.h>

Vector convertToVector(const Matrix& matrix) {
    if (matrix.m != 1) {
        throw SizeMismatchException();
    }

    DTYPE* vectorData = allocate1DArray(matrix.n);

    for (int i = 0; i < matrix.n; i++) {
        vectorData[i] = matrix(i, 0);
    }

    return Vector(vectorData, matrix.n);
}

Matrix convertToMatrix(const Vector& vector) {
    Matrix matrix = Matrix(vector.n, 1);

    for (int i = 0; i < vector.n; i++) {
        matrix(i, 0) = vector[i];
    }

    return matrix;
}