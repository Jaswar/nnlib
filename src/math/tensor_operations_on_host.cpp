/**
 * @file tensor_operations_on_host.cpp
 * @brief Source file defining tensor operations that happen on host.
 * @author Jan Warchocki
 * @date 29 August 2022
 */

#include "tensor_operations_on_host.h"
#include <cmath>

#if defined __AVX2__ || defined __AVX__
#include <immintrin.h>
#endif

#if defined __AVX2__ || defined __AVX__
/**
 * @brief Perform a horizontal add of a `__m256` value.
 *
 * This adds all 8 floats in such a value together.
 *
 * @param value The `__m256` variable whose floats should be summed up.
 * @return A float value corresponding to the sum.
 */
// value = [f7, f6, f5, f4, f3, f2, f1, f0]
float horizontalAdd(__m256 value) {
    // [f3, f2, f1, f0]
    auto low128 = _mm256_extractf128_ps(value, 0);

    // [f7, f6, f5, f4]
    auto high128 = _mm256_extractf128_ps(value, 1);

    // [f3 + f7, f2 + f6, f1 + f5, f0 + f4]
    __m128 sum128 = _mm_add_ps(low128, high128);

    // [f3 + f7, f2 + f6, f3 + f7, f2 + f6]
    __m128 sum128Moved = _mm_movehl_ps(sum128, sum128);

    // [dc, dc, f1 + f5 + f3 + f7, f0 + f4 + f2 + f6]
    __m128 sum128PlusMoved = _mm_add_ps(sum128, sum128Moved);

    // [dc, dc, f0 + f4 + f2 + f6, f1 + f5 + f3 + f7]
    auto shuffled = _mm_shuffle_ps(sum128PlusMoved, sum128PlusMoved, _MM_SHUFFLE(3, 2, 0, 1));

    // [dc, dc, dc, f1 + f5 + f3 + f7 + f0 + f4 + f2 + f6]
    __m128 final128Sum = _mm_add_ps(sum128PlusMoved, shuffled);

    return _mm_cvtss_f32(final128Sum);
}
#endif

float sumTensor(const Tensor& tensor) {
#if defined __AVX2__ || defined __AVX__
    __m256 accumulator = _mm256_setzero_ps();
    for (size_t index = 0; index < tensor.size / 8; index++) {
        const __m256 tensorValue = _mm256_loadu_ps(tensor.data + index * 8);
        accumulator = _mm256_add_ps(tensorValue, accumulator);
    }
    float accumulated = horizontalAdd(accumulator);

    for (size_t index = (tensor.size / 8) * 8; index < tensor.size; index++) {
        accumulated += tensor.data[index];
    }
    return accumulated;
#else
    float sum = 0;
    for (size_t i = 0; i < tensor.size; i++) {
        sum += tensor.data[i];
    }

    return sum;
#endif
}

void fillTensorOnHost(Tensor& tensor, float value) {
#if defined __AVX2__ || defined __AVX__
    __m256 valueVector = _mm256_set1_ps(value);
    for (size_t i = 0; i < tensor.size / 8; i++) {
        _mm256_storeu_ps(tensor.data + i * 8, valueVector);
    }
    for (size_t i = (tensor.size / 8) * 8; i < tensor.size; i++) {
        tensor.data[i] = value;
    }
#else
    for (size_t i = 0; i < tensor.size; i++) {
        tensor.data[i] = value;
    }
#endif
}

void addTensorsOnHost(const Tensor& a, const Tensor& b, Tensor& destination) {
#if defined __AVX2__ || defined __AVX__
    for (size_t index = 0; index < a.size / 8; index++) {
        __m256 am256 = _mm256_loadu_ps(a.data + index * 8);
        __m256 bm256 = _mm256_loadu_ps(b.data + index * 8);
        __m256 result = _mm256_add_ps(am256, bm256);
        _mm256_storeu_ps(destination.data + index * 8, result);
    }

    for (size_t index = (a.size / 8) * 8; index < a.size; index++) {
        destination.data[index] = a.data[index] + b.data[index];
    }
#else
    for (size_t i = 0; i < a.size; i++) {
        destination.data[i] = a.data[i] + b.data[i];
    }
#endif
}

void subtractTensorsOnHost(const Tensor& a, const Tensor& b, Tensor& destination) {
#if defined __AVX2__ || defined __AVX__
    for (size_t index = 0; index < a.size / 8; index++) {
        __m256 am256 = _mm256_loadu_ps(a.data + index * 8);
        __m256 bm256 = _mm256_loadu_ps(b.data + index * 8);
        __m256 result = _mm256_sub_ps(am256, bm256);
        _mm256_storeu_ps(destination.data + index * 8, result);
    }

    for (size_t index = (a.size / 8) * 8; index < a.size; index++) {
        destination.data[index] = a.data[index] - b.data[index];
    }
#else
    for (size_t i = 0; i < a.size; i++) {
        destination.data[i] = a.data[i] - b.data[i];
    }
#endif
}

void hadamardTensorsOnHost(const Tensor& a, const Tensor& b, Tensor& destination) {
#if defined __AVX2__ || defined __AVX__
    for (size_t index = 0; index < a.size / 8; index++) {
        __m256 am256 = _mm256_loadu_ps(a.data + index * 8);
        __m256 bm256 = _mm256_loadu_ps(b.data + index * 8);
        __m256 result = _mm256_mul_ps(am256, bm256);
        _mm256_storeu_ps(destination.data + index * 8, result);
    }

    for (size_t index = (a.size / 8) * 8; index < a.size; index++) {
        destination.data[index] = a.data[index] * b.data[index];
    }
#else
    for (size_t i = 0; i < a.size; i++) {
        destination.data[i] = a.data[i] * b.data[i];
    }
#endif
}


void divideTensorsOnHost(const Tensor& a, const Tensor& b, Tensor& destination) {
#if defined __AVX2__ || defined __AVX__
    for (size_t index = 0; index < a.size / 8; index++) {
        __m256 am256 = _mm256_loadu_ps(a.data + index * 8);
        __m256 bm256 = _mm256_loadu_ps(b.data + index * 8);
        __m256 result = _mm256_div_ps(am256, bm256);
        _mm256_storeu_ps(destination.data + index * 8, result);
    }

    for (size_t index = (a.size / 8) * 8; index < a.size; index++) {
        destination.data[index] = a.data[index] / b.data[index];
    }
#else
    for (size_t i = 0; i < a.size; i++) {
        destination.data[i] = a.data[i] / b.data[i];
    }
#endif
}

void logTensorOnHost(const Tensor& a, Tensor& destination) {
    for (size_t i = 0; i < a.size; i++) {
        destination.data[i] = logf(a.data[i]);
    }
}

void addBroadcastOnHost(const Tensor& matrix, const Tensor& vector, Tensor& destination) {
#if defined __AVX2__ || defined __AVX__
    for (size_t row = 0; row < matrix.shape[0]; row++) {
        for (size_t index = 0; index < matrix.shape[1] / 8; index++) {
            __m256 vectorData = _mm256_loadu_ps(vector.data + index * 8);
            __m256 matrixData = _mm256_loadu_ps(matrix.data + row * matrix.shape[1] + index * 8);
            __m256 res = _mm256_add_ps(vectorData, matrixData);
            _mm256_storeu_ps(destination.data + row * matrix.shape[1] + index * 8, res);
        }

        for (size_t index = (matrix.shape[1] / 8) * 8; index < matrix.shape[1]; index++) {
            destination.data[row * destination.shape[1] + index] =
                    matrix.data[row * matrix.shape[1] + index] + vector.data[index];
        }
    }
#else
    for (size_t i = 0; i < matrix.shape[0]; i++) {
        for (size_t j = 0; j < matrix.shape[1]; j++) {
            destination.data[i * destination.shape[1] + j] = matrix.data[i * matrix.shape[1] + j] + vector.data[j];
        }
    }
#endif
}

void multiplyTensorOnHost(const Tensor& tensor, float constant, Tensor& destination) {
#if defined __AVX2__ || defined __AVX__
    const __m256 constValue = _mm256_set1_ps(constant);
    for (size_t index = 0; index < tensor.size / 8; index++) {
        __m256 matrixValue = _mm256_loadu_ps(tensor.data + index * 8);
        __m256 result = _mm256_mul_ps(matrixValue, constValue);
        _mm256_storeu_ps(destination.data + index * 8, result);
    }

    for (size_t index = (tensor.size / 8) * 8; index < tensor.size; index++) {
        destination.data[index] = tensor.data[index] * constant;
    }
#else
    for (size_t i = 0; i < tensor.size; i++) {
        destination.data[i] = tensor.data[i] * constant;
    }
#endif
}

void multiplyMatrixVectorOnHost(const Tensor& matrix, const Tensor& vector, Tensor& destination) {
#if defined __AVX2__ || defined __AVX__
    for (size_t i = 0; i < matrix.shape[0]; i++) {
        float accumulator = 0;
        for (size_t index = 0; index < vector.shape[0] / 8; index++) {
            __m256 matrixData = _mm256_loadu_ps(matrix.data + i * matrix.shape[1] + index * 8);
            __m256 vectorData = _mm256_loadu_ps(vector.data + index * 8);
            accumulator += horizontalAdd(_mm256_mul_ps(matrixData, vectorData));
        }
        for (size_t index = (vector.shape[0] / 8) * 8; index < vector.shape[0]; index++) {
            accumulator += matrix.data[i * matrix.shape[1] + index] * vector.data[index];
        }
        destination.data[i] = accumulator;
    }
#else
    for (size_t i = 0; i < matrix.shape[0]; i++) {
        float accumulator = 0;
        for (size_t j = 0; j < vector.shape[0]; j++) {
            accumulator += matrix.data[i * matrix.shape[1] + j] * vector.data[j];
        }
        destination.data[i] = accumulator;
    }
#endif
}

/**
 * @brief Naive matrix-matrix multiplication.
 *
 * The method takes additional @p rowStart and @p columnStart arguments. These are used for edge cases in SIMD
 * implementation of matrix-matrix multiplication. They correspondingly signify which row and which column to start the
 * computation at.
 *
 * @param m1 The first operand matrix of the multiplication.
 * @param m2 The second operand matrix of the multiplication.
 * @param destination Where the result of the multiplication should be stored.
 * @param rowStart Specify which rows should be computed.
 * @param columnStart Specify which columns should be computed.
 */
void naiveMatMul(const Tensor& m1, const Tensor& m2, Tensor& destination, size_t rowStart = 0, size_t columnStart = 0) {
    size_t n = m1.shape[0];
    size_t k = m1.shape[1];
    size_t m = m2.shape[1];
    for (size_t row = rowStart; row < n; row++) {
        for (size_t column = columnStart; column < m; column++) {
            float acc = 0;
            for (size_t i = 0; i < k; i++) {
                acc += m1.data[row * k + i] * m2.data[i * m + column];
            }
            destination.data[row * m + column] = acc;
        }
    }
}

#if defined __AVX2__ || defined __AVX__
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
    const __m256 m1ColumnValue##index = _mm256_broadcast_ss(m1.data + (row * 8 + (index)) * k + i); \
    const __m256 mulResult##index = _mm256_mul_ps(m2Row, m1ColumnValue##index); \
    v##index = _mm256_add_ps(mulResult##index, v##index)

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
#define STORE_ROW(index) _mm256_storeu_ps(destination.data + (row * 8 + (index)) * m + column * 8, v##index)

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
void multiplyMatrixMatrixOnHost(const Tensor& m1, const Tensor& m2, Tensor& destination) {
#if defined __AVX2__ || defined __AVX__
    size_t n = m1.shape[0];
    size_t k = m1.shape[1];
    size_t m = m2.shape[1];
    for (size_t row = 0; row < n / 8; row++) {
        for (size_t column = 0; column < m / 8; column++) {
            SET_ALL_ROWS_TO_ZERO();

            for (size_t i = 0; i < k; i++) {
                // Load 8 floats from a row from m2
                const __m256 m2Row = _mm256_loadu_ps(m2.data + i * m + column * 8);

                COMPUTE_ALL_ROWS();
            }
            STORE_ALL_ROWS();
        }
    }

    naiveMatMul(m1, m2, destination, 0, (m / 8) * 8);
    naiveMatMul(m1, m2, destination, (n / 8) * 8, 0);
#else
    naiveMatMul(m1, m2, destination);
#endif
}

void transposeMatrixOnHost(const Tensor& matrix, Tensor& destination) {
    for (size_t i = 0; i < matrix.shape[0]; i++) {
        for (size_t j = 0; j < matrix.shape[1]; j++) {
            destination.data[j * destination.shape[1] + i] = matrix.data[i * matrix.shape[1] + j];
        }
    }
}
