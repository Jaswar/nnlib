//
// Created by Jan Warchocki on 29/05/2022.
//

#include "matrix_operations_on_host.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

void addMatricesOnHost(const Matrix& m1, const Matrix& m2, Matrix& result) {
#ifdef __AVX2__
    for (size_t index = 0; index < (m1.n * m1.m) / 8; index++) {
        __m256 a = _mm256_loadu_ps(m1.data + index * 8);
        __m256 b = _mm256_loadu_ps(m2.data + index * 8);
        __m256 res = _mm256_add_ps(a, b);
        _mm256_store_ps(result.data + index * 8, res);
    }

    for (size_t index = (m1.n * m1.m / 8) * 8; index < m1.n * m1.m; index++) {
        result.data[index] = m1.data[index] + m2.data[index];
    }
#else
    for (int i = 0; i < m1.n; i++) {
        for (int j = 0; j < m1.m; j++) {
            result(i, j) = m1(i, j) + m2(i, j);
        }
    }
#endif

}

void addBroadcastOnHost(const Matrix& m, const Vector& v, Matrix& result) {
#ifdef __AVX2__
    for (size_t row = 0; row < m.n; row++) {
        for (size_t index = 0; index < m.m / 8; index++) {
            __m256 vectorData = _mm256_loadu_ps(v.data + index * 8);
            __m256 matrixData = _mm256_loadu_ps(m.data + row * m.m + index * 8);
            __m256 res = _mm256_add_ps(vectorData, matrixData);
            _mm256_store_ps(result.data + row * m.m + index * 8, res);
        }

        for (size_t index = (m.m / 8) * 8; index < m.m; index++) {
            result(row, index) = m(row, index) + v[index];
        }
    }
#else
    for (int i = 0; i < m.n; i++) {
        for (int j = 0; j < m.m; j++) {
            result(i, j) = m(i, j) + v[j];
        }
    }
#endif
}

void subtractMatricesOnHost(const Matrix& m1, const Matrix& m2, Matrix& result) {
#ifdef __AVX2__
    for (size_t index = 0; index < (m1.n * m1.m) / 8; index++) {
        __m256 a = _mm256_loadu_ps(m1.data + index * 8);
        __m256 b = _mm256_loadu_ps(m2.data + index * 8);
        __m256 res = _mm256_sub_ps(a, b);
        _mm256_store_ps(result.data + index * 8, res);
    }

    for (size_t index = (m1.n * m1.m / 8) * 8; index < m1.n * m1.m; index++) {
        result.data[index] = m1.data[index] - m2.data[index];
    }
#else
    for (int i = 0; i < m1.n; i++) {
        for (int j = 0; j < m1.m; j++) {
            result(i, j) = m1(i, j) - m2(i, j);
        }
    }
#endif
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

void multiplyMatrixOnHost(const Matrix& m, DTYPE constant, Matrix& result) {
#ifdef __AVX2__
    __m256 constValue = _mm256_set1_ps(constant);
    for (size_t index = 0; index < (m.n * m.m) / 8; index++) {
        __m256 matrixValue = _mm256_loadu_ps(m.data + index * 8);
        __m256 res = _mm256_mul_ps(matrixValue, constValue);
        _mm256_store_ps(result.data + index * 8, res);
    }

    for (size_t index = (m.n * m.m / 8) * 8; index < m.n * m.m; index++) {
        result.data[index] = m.data[index] * constant;
    }
#else
    for (int i = 0; i < m.n; i++) {
        for (int j = 0; j < m.m; j++) {
            result(i, j) = m(i, j) * constant;
        }
    }
#endif
}

void hadamardMatricesOnHost(const Matrix& m1, const Matrix& m2, Matrix& result) {
#ifdef __AVX2__
    for (size_t index = 0; index < (m1.n * m1.m) / 8; index++) {
        __m256 a = _mm256_loadu_ps(m1.data + index * 8);
        __m256 b = _mm256_loadu_ps(m2.data + index * 8);
        __m256 res = _mm256_mul_ps(a, b);
        _mm256_store_ps(result.data + index * 8, res);
    }

    for (size_t index = (m1.n * m1.m / 8) * 8; index < m1.n * m1.m; index++) {
        result.data[index] = m1.data[index] * m2.data[index];
    }
#else
    for (int i = 0; i < m1.n; i++) {
        for (int j = 0; j < m1.m; j++) {
            result(i, j) = m1(i, j) * m2(i, j);
        }
    }
#endif
}

void transposeMatrixOnHost(const Matrix& m, Matrix& result) {
    for (int i = 0; i < m.n; i++) {
        for (int j = 0; j < m.m; j++) {
            result(j, i) = m(i, j);
        }
    }
}
