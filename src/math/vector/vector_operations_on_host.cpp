//
// Created by Jan Warchocki on 29/05/2022.
//

#include "vector_operations_on_host.h"
#ifdef __AVX2__
#include <immintrin.h>
#endif

void addVectorsOnHost(const Vector& v1, const Vector& v2, Vector& result) {
#ifdef __AVX2__
    for (size_t index = 0; index < v1.n / 8; index++) {
        __m256 a = _mm256_loadu_ps(v1.data + index * 8);
        __m256 b = _mm256_loadu_ps(v2.data + index * 8);
        __m256 res = _mm256_add_ps(a, b);
        _mm256_store_ps(result.data + index * 8, res);
    }

    for (size_t index = (v1.n / 8) * 8; index < v1.n; index++) {
        result[index] = v1[index] + v2[index];
    }
#else
    for (int i = 0; i < v1.n; i++) {
        result[i] = v1[i] + v2[i];
    }
#endif
}

void subtractVectorsOnHost(const Vector& v1, const Vector& v2, Vector& result) {
#ifdef __AVX2__
    for (size_t index = 0; index < v1.n / 8; index++) {
        __m256 a = _mm256_loadu_ps(v1.data + index * 8);
        __m256 b = _mm256_loadu_ps(v2.data + index * 8);
        __m256 res = _mm256_sub_ps(a, b);
        _mm256_store_ps(result.data + index * 8, res);
    }

    for (size_t index = (v1.n / 8) * 8; index < v1.n; index++) {
        result[index] = v1[index] - v2[index];
    }
#else
    for (int i = 0; i < v1.n; i++) {
        result[i] = v1[i] - v2[i];
    }
#endif
}

void multiplyVectorOnHost(const Vector& v1, DTYPE constant, Vector& result) {
#ifdef __AVX2__
    __m256 b = _mm256_set1_ps(constant);
    for (size_t index = 0; index < v1.n / 8; index++) {
        __m256 a = _mm256_loadu_ps(v1.data + index * 8);
        __m256 res = _mm256_mul_ps(a, b);
        _mm256_store_ps(result.data + index * 8, res);
    }

    for (size_t index = (v1.n / 8) * 8; index < v1.n; index++) {
        result[index] = v1[index] * constant;
    }
#else
    for (int i = 0; i < v1.n; i++) {
        result[i] = v1[i] * constant;
    }
#endif
}
