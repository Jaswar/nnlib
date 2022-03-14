//
// Created by Jan Warchocki on 03/03/2022.
//

#ifndef NNLIB_MATRIX_H
#define NNLIB_MATRIX_H

#include "../utils/allocation.h"
#include "../exceptions/size_mismatch_exception.h"
#include "vector.h"


class Matrix {
public:
    DTYPE* data;
    int n;
    int m;

    Matrix(int n, int m);
    Matrix(DTYPE* data, int n, int m);
    Matrix(const Matrix& matrix);

    ~Matrix();

    Matrix& operator=(const Matrix& matrix);
    DTYPE& operator()(int x, int y);
    DTYPE operator()(int x, int y) const;
};

// "toString" for std::cout
std::ostream& operator<<(std::ostream& stream, const Matrix& matrix);

Matrix operator+(const Matrix& m1, const Matrix& m2);
Matrix operator-(const Matrix& m1, const Matrix& m2);

Matrix operator*(const Matrix& m1, const Matrix& m2);
Vector operator*(const Matrix& m, const Vector& v);
Matrix operator*(const Matrix& m, DTYPE constant);
Matrix operator*(DTYPE constant, const Matrix& m);

#endif //NNLIB_MATRIX_H
