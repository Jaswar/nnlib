/**
 * @file tensor_operations_on_host.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 29 August 2022
 */

#include "tensor_operations_on_host.h"

#if defined __AVX2__ || defined __AVX__
#include <immintrin.h>
#endif

void addTensorsOnHost(const Tensor& a, const Tensor& b, Tensor& destination) {
#if defined __AVX2__ || defined __AVX__
    for (size_t index = 0; index < a.size / 8; index++) {
        __m256 am256 = _mm256_loadu_ps(a.host + index * 8);
        __m256 bm256 = _mm256_loadu_ps(b.host + index * 8);
        __m256 result = _mm256_add_ps(am256, bm256);
        _mm256_storeu_ps(destination.host + index * 8, result);
    }

    for (size_t index = (a.size / 8) * 8; index < a.size; index++) {
        destination.host[index] = a.host[index] + a.host[index];
    }
#else
    for (size_t i = 0; i < a.size; i++) {
        destination.host[i] = a.host[i] + b.host[i];
    }
#endif
}

void subtractTensorsOnHost(const Tensor& a, const Tensor& b, Tensor& destination) {
#if defined __AVX2__ || defined __AVX__
    for (size_t index = 0; index < a.size / 8; index++) {
        __m256 am256 = _mm256_loadu_ps(a.host + index * 8);
        __m256 bm256 = _mm256_loadu_ps(b.host + index * 8);
        __m256 result = _mm256_sub_ps(am256, bm256);
        _mm256_storeu_ps(destination.host + index * 8, result);
    }

    for (size_t index = (a.size / 8) * 8; index < a.size; index++) {
        destination.host[index] = a.host[index] - a.host[index];
    }
#else
    for (size_t i = 0; i < a.size; i++) {
        destination.host[i] = a.host[i] - b.host[i];
    }
#endif
}

void hadamardTensorsOnHost(const Tensor& a, const Tensor& b, Tensor& destination) {
#if defined __AVX2__ || defined __AVX__
    for (size_t index = 0; index < a.size / 8; index++) {
        __m256 am256 = _mm256_loadu_ps(a.host + index * 8);
        __m256 bm256 = _mm256_loadu_ps(b.host + index * 8);
        __m256 result = _mm256_mul_ps(am256, bm256);
        _mm256_storeu_ps(destination.host + index * 8, result);
    }

    for (size_t index = (a.size / 8) * 8; index < a.size; index++) {
        destination.host[index] = a.host[index] * a.host[index];
    }
#else
    for (size_t i = 0; i < a.size; i++) {
        destination.host[i] = a.host[i] * b.host[i];
    }
#endif
}

void multiplyTensorOnHost(const Tensor& tensor, float constant, Tensor& destination) {
#if defined __AVX2__f || defined __AVX__f
    const __m256 constValue = _mm256_set1_ps(constant);
    for (size_t index = 0; index < tensor.size / 8; index++) {
        __m256 matrixValue = _mm256_loadu_ps(tensor.host + index * 8);
        __m256 result = _mm256_mul_ps(matrixValue, constValue);
        _mm256_storeu_ps(destination.host + index * 8, result);
    }

    for (size_t index = (tensor.size / 8) * 8; index < tensor.size; index++) {
        destination.host[index] = tensor.host[index] * constant;
    }
#else
    for (size_t i = 0; i < tensor.size; i++) {
        destination.host[i] = tensor.host[i] * constant;
    }
#endif
}

void multiplyMatrixVectorOnHost(const Tensor& matrix, const Tensor& vector, Tensor& destination) {

}

void multiplyMatrixMatrixOnHost(const Tensor& m1, const Tensor& m2, Tensor& destination) {

}

void transposeMatrixOnHost(const Tensor& matrix, Tensor& destination) {

}
