//
// Created by Jan Warchocki on 03/03/2022.
//
#include "matrix.h"

Matrix::Matrix(int n, int m) : Matrix(allocate2DArray(n, m), n, m) {}

Matrix::Matrix(DTYPE** data, int n, int m) : data(data), n(n), m(m) {}

Matrix::Matrix(const Matrix& matrix) {
    n = matrix.n;
    m = matrix.m;

    data = copy2DArray(n, m, matrix.data);
}

Matrix::~Matrix() {
    for (int i = 0; i < n; i++) {
        free(data[i]);
    }

    free(data);
}

Matrix& Matrix::operator=(const Matrix& matrix) {
    for (int i = 0; i < n; i++) {
        free(data[i]);
    }

    free(data);

    n = matrix.n;
    m = matrix.m;

    data = copy2DArray(n, m, matrix.data);

    return *this;
}

DTYPE*& Matrix::operator[](int index) {
    return data[index];
}

DTYPE* Matrix::operator[](int index) const {
    return data[index];
}

void displayRow(std::ostream& stream, DTYPE* row, int m) {
    stream << "[";

    for (int i = 0; i < m - 1; i++) {
        stream << row[i] << ", ";
    }

    if (m > 0) {
        stream << row[m - 1];
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
            displayRow(stream, matrix[i], matrix.m);

            stream << ",\n";
            stream << "\t";
        } else if (i == maxPeek) {
            stream << "[... " << matrix.n - maxPeek * 2 << " more rows ...]\n\t";
        }
    }

    if (matrix.n > 0) {
        displayRow(stream, matrix[matrix.n - 1], matrix.m);
    }

    stream << "])";

    return stream;
}

Matrix operator+(const Matrix& m1, const Matrix& m2) {

    if (m1.n != m2.n || m1.m != m2.m) {
        throw SizeMismatchException();
    }

    DTYPE** newData = allocate2DArray(m1.n, m1.m);

    for (int i = 0; i < m1.n; i++) {
        for (int j = 0; j < m1.m; j++) {
            newData[i][j] = m1[i][j] + m2[i][j];
        }
    }

    return Matrix(newData, m1.n, m1.m);
}

Matrix operator-(const Matrix& m1, const Matrix& m2) {
    if (m1.n != m2.n || m1.m != m2.m) {
        throw SizeMismatchException();
    }

    DTYPE** newData = allocate2DArray(m1.n, m1.m);

    for (int i = 0; i < m1.n; i++) {
        for (int j = 0; j < m1.m; j++) {
            newData[i][j] = m1[i][j] - m2[i][j];
        }
    }

    return Matrix(newData, m1.n, m1.m);
}

Matrix operator*(const Matrix& m1, const Matrix& m2) {
    if (m1.m != m2.n) {
        throw SizeMismatchException();
    }

    DTYPE** newData = allocate2DArray(m1.n, m2.m);

    for (int row = 0; row < m1.n; row++) {
        for (int column = 0; column < m2.m; column++) {

            DTYPE sum = 0;
            for (int i = 0; i < m1.m; i++) {
                sum += m1[row][i] * m2[i][column];
            }
            newData[row][column] = sum;
        }
    }

    return Matrix(newData, m1.n, m2.m);
}

Vector operator*(const Matrix& m, const Vector& v) {
    if (m.m != v.n) {
        throw SizeMismatchException();
    }

    DTYPE* newData = allocate1DArray(m.n);

    for (int i = 0; i < m.n; i++) {
        newData[i] = 0;
        for (int j = 0; j < v.n; j++) {
            newData[i] += m[i][j] * v[j];
        }
    }

    return Vector(newData, m.n);
}

Matrix operator*(const Matrix& m, DTYPE constant) {
    DTYPE** newData = allocate2DArray(m.n, m.m);

    for (int i = 0; i < m.n; i++) {
        for (int j = 0; j < m.m; j++) {
            newData[i][j] = m[i][j] * constant;
        }
    }

    return Matrix(newData, m.n, m.m);
}

Matrix operator*(DTYPE constant, const Matrix& m2) {
    return m2 * constant;
}
