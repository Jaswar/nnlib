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
    dLocation location;

    Matrix(int n, int m);
    Matrix(int n, int m, dLocation location);
    Matrix(DTYPE* data, int n, int m);
    Matrix(DTYPE* data, int n, int m, dLocation location);
    Matrix(const Matrix& matrix);

    void moveToDevice();
    void moveToHost();

    ~Matrix();

    Matrix& operator=(const Matrix& matrix);
    DTYPE& operator()(int x, int y);
    DTYPE operator()(int x, int y) const;
};

// "toString" for std::cout
std::ostream& operator<<(std::ostream& stream, const Matrix& matrix);

void add(const Matrix& m1, const Matrix& m2, Matrix& result);
void subtract(const Matrix& m1, const Matrix& m2, Matrix& result);

void multiply(const Matrix& m, const Vector& v, Vector& result);
void multiply(const Matrix& m, DTYPE constant, Matrix& result);
void multiply(DTYPE constant, const Matrix& m, Matrix& result);

#endif //NNLIB_MATRIX_H
