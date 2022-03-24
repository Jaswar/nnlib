//
// Created by Jan Warchocki on 03/03/2022.
//
#include "matrix.h"
#include "../gpu/allocation_gpu.cuh"
#include "../exceptions/different_data_location_exception.h"
#include "matrix_operations.cuh"

Matrix::Matrix(int n, int m) : Matrix(n, m, HOST) {}

Matrix::Matrix(int n, int m, dLocation location) : n(n), m(m), location(location) {
    if (location == HOST) {
        if (n > 0 && m > 0) {
            this->data = allocate1DArray(n * m);
        } else {
            this->data = nullptr;
        }
    } else {
        if (n > 0 && m > 0) {
            this->data = allocate1DArrayDevice(n * m);
        } else {
            this->data = nullptr;
        }
    }
}

Matrix::Matrix(DTYPE* data, int n, int m) : Matrix(data, n, m, HOST) {}

Matrix::Matrix(DTYPE* data, int n, int m, dLocation location) : data(data), n(n), m(m), location(location) {}

Matrix::Matrix(const Matrix& matrix) {
    location = matrix.location;
    n = matrix.n;
    m = matrix.m;

    if (location == HOST) {
        data = copy1DArray(n * m, matrix.data);
    } else {
        data = copy1DArrayDevice(n * m, matrix.data);
    }
}

void Matrix::moveToDevice() {
    if (location == DEVICE) {
        return;
    }

    DTYPE* deviceData = allocate1DArrayDevice(n * m);
    copy1DFromHostToDevice(data, deviceData, n * m);

    free(data);
    data = deviceData;
    location = DEVICE;
}

void Matrix::moveToHost() {
    if (location == HOST) {
        return;
    }

    DTYPE* hostData = allocate1DArray(n * m);
    copy1DFromDeviceToHost(data, hostData, n * m);

    free1DArrayDevice(data);
    data = hostData;
    location = HOST;
}

Matrix::~Matrix() {
    if (n == 0 || m == 0) {
        return;
    }

    if (location == HOST) {
        free(data);
    } else {
        free1DArrayDevice(data);
    }
}

Matrix& Matrix::operator=(const Matrix& matrix) {
    location = matrix.location;

    if (location == HOST) {
        if (n > 0 && m > 0) {
            free(data);
        }
        data = copy1DArray(matrix.n * matrix.m, matrix.data);
    } else {
        if (n > 0 && m > 0) {
            free1DArrayDevice(data);
        }
        data = copy1DArrayDevice(matrix.n * matrix.m, matrix.data);
    }

    n = matrix.n;
    m = matrix.m;

    return *this;
}

DTYPE& Matrix::operator()(int x, int y) {
    return data[x * m + y];
}

DTYPE Matrix::operator()(int x, int y) const {
    return data[x * m + y];
}

void displayRow(std::ostream& stream, const Matrix& matrix, int row, int m) {
    stream << "[";

    for (int i = 0; i < m - 1; i++) {
        stream << matrix(row, i) << ", ";
    }

    if (m > 0) {
        stream << matrix(row, m - 1);
    }

    stream << "]";
}

std::ostream& operator<<(std::ostream& stream, const Matrix& matrix) {
    int maxPeek = 10; // Number of lines to show from the front and back of matrix.

    stream << "Matrix([";

    for (int i = 0; i < matrix.n - 1; i++) {
        // Make sure no more than 50 first and 50 last rows are displayed
        // to avoid flooding the console.
        if (i < maxPeek || i >= matrix.n - maxPeek) {
            displayRow(stream, matrix, i, matrix.m);

            stream << ",\n";
            stream << "\t";
        } else if (i == maxPeek) {
            stream << "[... " << matrix.n - maxPeek * 2 << " more rows ...]\n\t";
        }
    }

    if (matrix.n > 0) {
        displayRow(stream, matrix, matrix.n - 1, matrix.m);
    }

    stream << "])";

    return stream;
}

void add(const Matrix& m1, const Matrix& m2, Matrix& result) {
    if (m1.n != m2.n || m1.m != m2.m
        || m1.n != result.n || m1.m != result.m
        || m2.n != result.n || m2.m != result.m ) {
        throw SizeMismatchException();
    }
    if (m1.location != m2.location || m1.location != result.location || m2.location != result.location) {
        throw DifferentDataLocationException();
    }

    if (m1.location == HOST) {
        for (int i = 0; i < m1.n; i++) {
            for (int j = 0; j < m1.m; j++) {
                result(i, j) = m1(i, j) + m2(i, j);
            }
        }
    } else {
        addMatrices(m1, m2, result);
    }
}

void add(const Matrix& m, const Vector& v, Matrix& result) {
    if (m.m != v.n || m.m != result.m || m.n != result.n || result.m != v.n) {
        throw SizeMismatchException();
    }
    if (m.location != v.location || v.location != result.location || m.location != result.location) {
        throw DifferentDataLocationException();
    }

    if (m.location == HOST) {
        for (int i = 0; i < m.n; i++) {
            for (int j = 0; j < m.m; j++) {
                result(i, j) = m(i, j) + v[j];
            }
        }
    } else {
        addBroadcast(m, v, result);
    }
}

void subtract(const Matrix& m1, const Matrix& m2, Matrix& result) {
    if (m1.n != m2.n || m1.m != m2.m
        || m1.n != result.n || m1.m != result.m
        || m2.n != result.n || m2.m != result.m ) {
        throw SizeMismatchException();
    }
    if (m1.location != m2.location || m1.location != result.location || m2.location != result.location) {
        throw DifferentDataLocationException();
    }

    if (m1.location == HOST) {
        for (int i = 0; i < m1.n; i++) {
            for (int j = 0; j < m1.m; j++) {
                result(i, j) = m1(i, j) - m2(i, j);
            }
        }
    } else {
        subtractMatrices(m1, m2, result);
    }
}

void multiply(const Matrix& m1, const Matrix& m2, Matrix& result) {
    if (m1.m != m2.n || m1.n != result.n || m2.m != result.m) {
        throw SizeMismatchException();
    }
    if (m1.location != m2.location || m1.location != result.location || m2.location != result.location) {
        throw DifferentDataLocationException();
    }

    if (m1.location == HOST) {
        for (int row = 0; row < m1.n; row++) {
            for (int column = 0; column < m2.m; column++) {
                DTYPE sum = 0;
                for (int i = 0; i < m1.m; i++) {
                    sum += m1(row, i) * m2(i, column);
                }
                result(row, column) = sum;
            }
        }
    } else {
        multiplyMatrices(m1, m2, result);
    }
}

void multiply(const Matrix& m, const Vector& v, Vector& result) {
    if (m.m != v.n || result.n != m.n) {
        throw SizeMismatchException();
    }
    if (m.location != v.location || m.location != result.location || v.location != result.location) {
        throw DifferentDataLocationException();
    }

    if (m.location == HOST) {
        for (int i = 0; i < m.n; i++) {
            result[i] = 0;
            for (int j = 0; j < v.n; j++) {
                result[i] += m(i, j) * v[j];
            }
        }
    } else {
        multiplyMatrixVector(m, v, result);
    }
}

void multiply(const Matrix& m, DTYPE constant, Matrix& result) {
    if (m.n != result.n || m.m != result.m) {
        throw SizeMismatchException();
    }
    if (m.location != result.location) {
        throw DifferentDataLocationException();
    }

    if (m.location == HOST) {
        for (int i = 0; i < m.n; i++) {
            for (int j = 0; j < m.m; j++) {
                result(i, j) = m(i, j) * constant;
            }
        }
    } else {
        multiplyMatrix(m, constant, result);
    }
}

void multiply(DTYPE constant, const Matrix& m, Matrix& result) {
    multiply(m, constant, result);
}

void transpose(const Matrix& m, Matrix& result) {
    if (m.n != result.m || m.m != result.n) {
        throw SizeMismatchException();
    }
    if (m.location != result.location) {
        throw DifferentDataLocationException();
    }

    if (m.location == HOST) {
        for (int i = 0; i < m.n; i++) {
            for (int j = 0; j < m.m; j++) {
                result(j, i) = m(i, j);
            }
        }
    } else {
        transposeMatrix(m, result);
    }
}