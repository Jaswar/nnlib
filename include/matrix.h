//
// Created by Jan Warchocki on 03/03/2022.
//

#ifndef NNLIB_MATRIX_H
#define NNLIB_MATRIX_H


#include "vector.h"


class Matrix {
public:
    DTYPE* data;
    size_t n;
    size_t m;
    DataLocation location;

    Matrix(size_t n, size_t m);
    Matrix(size_t n, size_t m, DataLocation location);
    Matrix(DTYPE* data, size_t n, size_t m);
    Matrix(DTYPE* data, size_t n, size_t m, DataLocation location);
    Matrix(const Matrix& matrix);

    void moveToDevice();
    void moveToHost();

    ~Matrix();

    Matrix& operator=(const Matrix& matrix);

    DTYPE& operator()(size_t x, size_t y);
    DTYPE operator()(size_t x, size_t y) const;
};

// "toString" for std::cout
std::ostream& operator<<(std::ostream& stream, const Matrix& matrix);

void add(const Matrix& m1, const Matrix& m2, Matrix& result);

// Broadcast v onto m
void add(const Matrix& m, const Vector& v, Matrix& result);

void subtract(const Matrix& m1, const Matrix& m2, Matrix& result);

void multiply(const Matrix& m1, const Matrix& m2, Matrix& result);
void multiply(const Matrix& m, const Vector& v, Vector& result);
void multiply(const Matrix& m, DTYPE constant, Matrix& result);
void multiply(DTYPE constant, const Matrix& m, Matrix& result);

void hadamard(const Matrix& m1, const Matrix& m2, Matrix& result);

void transpose(const Matrix& m, Matrix& result);

#endif //NNLIB_MATRIX_H
