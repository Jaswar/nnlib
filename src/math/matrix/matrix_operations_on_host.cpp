/**
 * @file matrix_operations_on_host.cpp
 * @brief Source file defining matrix operations that happen on host.
 * @author Jan Warchocki
 * @date 29 May 2022
 */

#include "matrix_operations_on_host.h"

#if defined __AVX2__ || defined __AVX__
#include <immintrin.h>
#endif

void addMatricesOnHost(const Matrix& m1, const Matrix& m2, Matrix& result) {
#if defined __AVX2__ || defined __AVX__
    for (size_t index = 0; index < (m1.n * m1.m) / 8; index++) {
        __m256 a = _mm256_loadu_ps(m1.data + index * 8);
        __m256 b = _mm256_loadu_ps(m2.data + index * 8);
        __m256 res = _mm256_add_ps(a, b);
        _mm256_storeu_ps(result.data + index * 8, res);
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
#if defined __AVX2__ || defined __AVX__
    for (size_t row = 0; row < m.n; row++) {
        for (size_t index = 0; index < m.m / 8; index++) {
            __m256 vectorData = _mm256_loadu_ps(v.data + index * 8);
            __m256 matrixData = _mm256_loadu_ps(m.data + row * m.m + index * 8);
            __m256 res = _mm256_add_ps(vectorData, matrixData);
            _mm256_storeu_ps(result.data + row * m.m + index * 8, res);
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
#if defined __AVX2__ || defined __AVX__
    for (size_t index = 0; index < (m1.n * m1.m) / 8; index++) {
        __m256 a = _mm256_loadu_ps(m1.data + index * 8);
        __m256 b = _mm256_loadu_ps(m2.data + index * 8);
        __m256 res = _mm256_sub_ps(a, b);
        _mm256_storeu_ps(result.data + index * 8, res);
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

#if defined __AVX2__ || defined __AVX__
/**
 * @brief Perform a horizontal add of a `__m256` value.
 *
 * This adds all 8 floats in such a value together.
 *
 * @param value The `__m256` variable whose floats should be summed up.
 * @return A float value corresponding to the sum.
 */
//// value = [f7, f6, f5, f4, f3, f2, f1, f0]
//float horizontalAdd(__m256 value) {
//    // [f3, f2, f1, f0]
//    auto low128 = _mm256_extractf128_ps(value, 0);
//
//    // [f7, f6, f5, f4]
//    auto high128 = _mm256_extractf128_ps(value, 1);
//
//    // [f3 + f7, f2 + f6, f1 + f5, f0 + f4]
//    __m128 sum128 = _mm_add_ps(low128, high128);
//
//    // [f3 + f7, f2 + f6, f3 + f7, f2 + f6]
//    __m128 sum128Moved = _mm_movehl_ps(sum128, sum128);
//
//    // [dc, dc, f1 + f5 + f3 + f7, f0 + f4 + f2 + f6]
//    __m128 sum128PlusMoved = _mm_add_ps(sum128, sum128Moved);
//
//    // [dc, dc, f0 + f4 + f2 + f6, f1 + f5 + f3 + f7]
//    auto shuffled = _mm_shuffle_ps(sum128PlusMoved, sum128PlusMoved, _MM_SHUFFLE(3, 2, 0, 1));
//
//    // [dc, dc, dc, f1 + f5 + f3 + f7 + f0 + f4 + f2 + f6]
//    __m128 final128Sum = _mm_add_ps(sum128PlusMoved, shuffled);
//
//    return _mm_cvtss_f32(final128Sum);
//}
#endif

void multiplyMatrixVectorOnHost(const Matrix& matrix, const Vector& vector, Vector& result) {
#if defined __AVX2__ || defined __AVX__
    for (size_t i = 0; i < matrix.n; i++) {
        float accumulator = 0;
        for (size_t index = 0; index < vector.n / 8; index++) {
            __m256 matrixData = _mm256_loadu_ps(matrix.data + i * matrix.m + index * 8);
            __m256 vectorData = _mm256_loadu_ps(vector.data + index * 8);
//            accumulator += horizontalAdd(_mm256_mul_ps(matrixData, vectorData));
        }
        for (size_t index = (vector.n / 8) * 8; index < vector.n; index++) {
            accumulator += matrix.data[i * matrix.m + index] * vector.data[index];
        }
        result.data[i] = accumulator;
    }
#else
    for (int i = 0; i < matrix.n; i++) {
        result[i] = 0;
        for (int j = 0; j < vector.n; j++) {
            result[i] += matrix(i, j) * vector[j];
        }
    }
#endif
}

#if defined __AVX2__ || defined __AVX__
/**
 * @brief Handle edge cases when performing matrix-matrix multiplication with AVX2 instructions.
 *
 * The AVX2 matrix multiplication algorithm works by computing 8x8 tiles at a time. If the size of the result
 * matrix is not divisible by 8, there are leftover elements that should be computed. These edge case elements
 * are computed using this method.
 *
 * To make sure only the edge case elements are computed, @p rowStart and @p columnStart should be specified.
 * These parameters specify the range where the edge case elements can be found. Computation will then only happen in
 * the range: `[rowStart, result.n] x [columnStart, result.m]`
 *
 * @param m1 The first operand matrix of the multiplication.
 * @param m2 The second operand matrix of the multiplication.
 * @param result Where the result of the multiplication should be stored.
 * @param rowStart Specify which rows should be computed.
 * @param columnStart Specify which columns should be computed.
 */
void handleAVX2MatMulEdgeCases(const Matrix& m1, const Matrix& m2, Matrix& result, size_t rowStart,
                               size_t columnStart) {
    for (size_t row = rowStart; row < m1.n; row++) {
        for (size_t column = columnStart; column < m2.m; column++) {
            float acc = 0;
            for (size_t k = 0; k < m2.n; k++) {
                acc += m1.data[row * m1.m + k] * m2.data[k * m2.m + column];
            }
            result.data[row * result.m + column] = acc;
        }
    }
}

/**
 * @brief Set a row of an 8x8 tile to 0.
 */
#define SET_ROW_TO_ZERO(index) __m256 v##index = _mm256_setzero_ps()

/**
 * @brief Set all rows of an 8x8 tile to 0.
 */
#define SET_ALL_ROWS_TO_ZERO() \
    SET_ROW_TO_ZERO(0); \
    SET_ROW_TO_ZERO(1); \
    SET_ROW_TO_ZERO(2); \
    SET_ROW_TO_ZERO(3); \
    SET_ROW_TO_ZERO(4); \
    SET_ROW_TO_ZERO(5); \
    SET_ROW_TO_ZERO(6); \
    SET_ROW_TO_ZERO(7)

/**
 * @brief Compute a single row of an 8x8 tile.
 */
#define COMPUTE_ROW(index) \
    m1ColumnValue = _mm256_broadcast_ss(m1.data + (row * 8 + (index)) * m1.m + k); \
    mulResult = _mm256_mul_ps(m2Row, m1ColumnValue); \
    v##index = _mm256_add_ps(mulResult, v##index)

/**
 * @brief Compute all rows of an 8x8 tile.
 */
#define COMPUTE_ALL_ROWS() \
    COMPUTE_ROW(0); \
    COMPUTE_ROW(1); \
    COMPUTE_ROW(2); \
    COMPUTE_ROW(3); \
    COMPUTE_ROW(4); \
    COMPUTE_ROW(5); \
    COMPUTE_ROW(6); \
    COMPUTE_ROW(7)

/**
 * @brief Store a row of 8x8 tile to the result matrix.
 */
#define STORE_ROW(index) _mm256_storeu_ps(result.data + (row * 8 + (index)) * result.m + column * 8, v##index)

/**
 * @brief Store all rows of an 8x8 tile to the result matrix.
 */
#define STORE_ALL_ROWS() \
    STORE_ROW(0); \
    STORE_ROW(1); \
    STORE_ROW(2); \
    STORE_ROW(3); \
    STORE_ROW(4); \
    STORE_ROW(5); \
    STORE_ROW(6); \
    STORE_ROW(7)

#endif

// Based on https://github.com/yzhaiustc/Optimizing-DGEMM-on-Intel-CPUs-with-AVX512F/blob/master/include/kernel5.h
// Disable linter about the number of lines. That's the tradeoff for a slightly faster method.
// NOLINTNEXTLINE(google-readability-function-size)
void multiplyMatricesOnHost(const Matrix& m1, const Matrix& m2, Matrix& result) {
#if defined __AVX2__ || defined __AVX__
    for (size_t row = 0; row < m1.n / 8; row++) {
        for (size_t column = 0; column < m2.m / 8; column++) {
            SET_ALL_ROWS_TO_ZERO();

            __m256 m1ColumnValue;
            __m256 mulResult;
            for (size_t k = 0; k < m2.n; k++) {
                // Load 8 floats from a row from m2
                const __m256 m2Row = _mm256_loadu_ps(m2.data + k * m2.m + column * 8);

                COMPUTE_ALL_ROWS();
            }
            STORE_ALL_ROWS();
        }
    }

    handleAVX2MatMulEdgeCases(m1, m2, result, 0, (m2.m / 8) * 8);
    handleAVX2MatMulEdgeCases(m1, m2, result, (m1.n / 8) * 8, 0);
#else
    for (int row = 0; row < m1.n; row++) {
        for (int column = 0; column < m2.m; column++) {
            DTYPE sum = 0;
            for (int i = 0; i < m1.m; i++) {
                sum += m1(row, i) * m2(i, column);
            }
            result(row, column) = sum;
        }
    }
#endif
}

void multiplyMatrixOnHost(const Matrix& m, DTYPE constant, Matrix& result) {
#if defined __AVX2__ || defined __AVX__
    __m256 constValue = _mm256_set1_ps(constant);
    for (size_t index = 0; index < (m.n * m.m) / 8; index++) {
        __m256 matrixValue = _mm256_loadu_ps(m.data + index * 8);
        __m256 res = _mm256_mul_ps(matrixValue, constValue);
        _mm256_storeu_ps(result.data + index * 8, res);
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
#if defined __AVX2__ || defined __AVX__
    for (size_t index = 0; index < (m1.n * m1.m) / 8; index++) {
        __m256 a = _mm256_loadu_ps(m1.data + index * 8);
        __m256 b = _mm256_loadu_ps(m2.data + index * 8);
        __m256 res = _mm256_mul_ps(a, b);
        _mm256_storeu_ps(result.data + index * 8, res);
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
