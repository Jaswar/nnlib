//
// Created by Jan Warchocki on 03/03/2022.
//
#include "matrix.h"
#include "../gpu/allocation_gpu.cuh"
#include "../exceptions/different_data_location_exception.h"
#include "../gpu/matrix_operations.cuh"

Matrix::Matrix(int n, int m) : Matrix(allocate1DArray(n * m), n, m, HOST) {}

Matrix::Matrix(int n, int m, dLocation location) : n(n), m(m), location(location) {
    if (location == HOST) {
        this->data = allocate1DArray(n * m);
    } else {
        this->data = allocate1DArrayDevice(n * m);
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
    if (location == HOST) {
        free(data);
    } else {
        free1DArrayDevice(data);
    }
}

Matrix& Matrix::operator=(const Matrix& matrix) {
    n = matrix.n;
    m = matrix.m;
    location = matrix.location;

    if (location == HOST) {
        free(data);
        data = copy1DArray(n * m, matrix.data);
    } else {
        free1DArrayDevice(data);
        data = copy1DArrayDevice(n * m, matrix.data);
    }

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

Matrix operator+(const Matrix& m1, const Matrix& m2) {
    if (m1.n != m2.n || m1.m != m2.m) {
        throw SizeMismatchException();
    }
    if (m1.location != m2.location) {
        throw DifferentDataLocationException();
    }

    if (m1.location == HOST) {
        Matrix result = Matrix(m1.n, m1.m);

        for (int i = 0; i < m1.n; i++) {
            for (int j = 0; j < m1.m; j++) {
                result(i, j) = m1(i, j) + m2(i, j);
            }
        }

        return result;
    } else {
        return addMatrices(m1, m2);
    }
}

Matrix operator-(const Matrix& m1, const Matrix& m2) {
    if (m1.n != m2.n || m1.m != m2.m) {
        throw SizeMismatchException();
    }
    if (m1.location != m2.location) {
        throw DifferentDataLocationException();
    }

    if (m1.location == HOST) {
        Matrix result = Matrix(m1.n, m1.m);

        for (int i = 0; i < m1.n; i++) {
            for (int j = 0; j < m1.m; j++) {
                result(i, j) = m1(i, j) - m2(i, j);
            }
        }

        return result;
    } else {
        return subtractMatrices(m1, m2);
    }
}

//Matrix operator*(const Matrix& m1, const Matrix& m2) {
//    if (m1.m != m2.n) {
//        throw SizeMismatchException();
//    }
//
//    Matrix result = Matrix(m1.n, m2.m);
//
//    for (int row = 0; row < m1.n; row++) {
//        for (int column = 0; column < m2.m; column++) {
//
//            DTYPE sum = 0;
//            for (int i = 0; i < m1.m; i++) {
//                sum += m1(row, i) * m2(i, column);
//            }
//            result(row, column) = sum;
//        }
//    }
//
//    return result;
//}

Vector operator*(const Matrix& m, const Vector& v) {
    if (m.m != v.n) {
        throw SizeMismatchException();
    }
    if (m.location != v.location) {
        throw DifferentDataLocationException();
    }

    if (m.location == HOST) {
        DTYPE* newData = allocate1DArray(m.n);

        for (int i = 0; i < m.n; i++) {
            newData[i] = 0;
            for (int j = 0; j < v.n; j++) {
                newData[i] += m(i, j) * v[j];
            }
        }

        return Vector(newData, m.n);
    } else {
        return multiplyMatrixVector(m, v);
    }
}

Matrix operator*(const Matrix& m, DTYPE constant) {
    if (m.location == HOST) {
        Matrix result = Matrix(m.n, m.m);

        for (int i = 0; i < m.n; i++) {
            for (int j = 0; j < m.m; j++) {
                result(i, j) = m(i, j) * constant;
            }
        }

        return result;
    } else {
        return multiplyMatrix(m, constant);
    }
}

Matrix operator*(DTYPE constant, const Matrix& m2) {
    return m2 * constant;
}
