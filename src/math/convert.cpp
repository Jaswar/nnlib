//
// Created by Jan Warchocki on 07/03/2022.
//

#include <exceptions/size_mismatch_exception.h>
#include "convert.h"

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