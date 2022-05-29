//
// Created by Jan Warchocki on 29/05/2022.
//

#include "matrix_operations_on_host.h"

void addMatricesOnHost(const Matrix& m1, const Matrix& m2, Matrix& result) {
    for (int i = 0; i < m1.n; i++) {
        for (int j = 0; j < m1.m; j++) {
            result(i, j) = m1(i, j) + m2(i, j);
        }
    }
}

void addBroadcastOnHost(const Matrix& m, const Vector& v, Matrix& result) {
    for (int i = 0; i < m.n; i++) {
        for (int j = 0; j < m.m; j++) {
            result(i, j) = m(i, j) + v[j];
        }
    }
}

void subtractMatricesOnHost(const Matrix& m1, const Matrix& m2, Matrix& result) {
    for (int i = 0; i < m1.n; i++) {
        for (int j = 0; j < m1.m; j++) {
            result(i, j) = m1(i, j) - m2(i, j);
        }
    }
}

void multiplyMatrixVectorOnHost(const Matrix& m, const Vector& v, Vector& result) {
    for (int i = 0; i < m.n; i++) {
        result[i] = 0;
        for (int j = 0; j < v.n; j++) {
            result[i] += m(i, j) * v[j];
        }
    }
}

void multiplyMatricesOnHost(const Matrix& m1, const Matrix& m2, Matrix& result) {
    for (int row = 0; row < m1.n; row++) {
        for (int column = 0; column < m2.m; column++) {
            DTYPE sum = 0;
            for (int i = 0; i < m1.m; i++) {
                sum += m1(row, i) * m2(i, column);
            }
            result(row, column) = sum;
        }
    }
}

void multiplyMatrixOnHost(const Matrix& m, float constant, Matrix& result) {
    for (int i = 0; i < m.n; i++) {
        for (int j = 0; j < m.m; j++) {
            result(i, j) = m(i, j) * constant;
        }
    }
}

void hadamardMatricesOnHost(const Matrix& m1, const Matrix& m2, Matrix& result) {
    for (int i = 0; i < m1.n; i++) {
        for (int j = 0; j < m1.m; j++) {
            result(i, j) = m1(i, j) * m2(i, j);
        }
    }
}

void transposeMatrixOnHost(const Matrix& m, Matrix& result) {
    for (int i = 0; i < m.n; i++) {
        for (int j = 0; j < m.m; j++) {
            result(j, i) = m(i, j);
        }
    }
}
