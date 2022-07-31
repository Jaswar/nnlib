//
// Created by Jan Warchocki on 29/05/2022.
//

#include <layer.h>

#if defined __AVX2__ || defined __AVX__
#include <immintrin.h>
#endif

#if defined __AVX2__ || defined __AVX__
void applyBiasGradients(Layer& layer, size_t batchSize, DTYPE learningRate) {
    const __m256 constant = _mm256_set1_ps(learningRate / static_cast<float>(batchSize));

    for (size_t i = 0; i < layer.outSize / 8; i++) {
        const __m256 oldBiases = _mm256_loadu_ps(layer.biases.data + i * 8);
        const __m256 biasesGradients = _mm256_loadu_ps(layer.biasesGradients.data + i * 8);
        const __m256 newBiases = _mm256_sub_ps(oldBiases, _mm256_mul_ps(constant, biasesGradients));

        _mm256_storeu_ps(layer.biases.data + i * 8, newBiases);
        _mm256_storeu_ps(layer.biasesGradients.data + i * 8, _mm256_setzero_ps());
    }

    for (size_t i = (layer.outSize / 8) * 8; i < layer.outSize; i++) {
        layer.biases[i] -= learningRate * layer.biasesGradients[i] / static_cast<DTYPE>(batchSize);
        layer.biasesGradients[i] = 0;
    }
}

void applyWeightGradients(Layer& layer, size_t batchSize, DTYPE learningRate) {
    const __m256 constant = _mm256_set1_ps(learningRate / static_cast<float>(batchSize));

    for (size_t i = 0; i < layer.inSize; i++) {
        for (size_t j = 0; j < layer.outSize / 8; j++) {
            const __m256 oldWeights = _mm256_loadu_ps(layer.weights.data + i * layer.outSize + j * 8);
            const __m256 weightsGradients = _mm256_loadu_ps(layer.weightsGradients.data + i * layer.outSize + j * 8);
            const __m256 newWeights = _mm256_sub_ps(oldWeights, _mm256_mul_ps(constant, weightsGradients));

            _mm256_storeu_ps(layer.weights.data + i * layer.outSize + j * 8, newWeights);
            _mm256_storeu_ps(layer.weightsGradients.data + i * layer.outSize + j * 8, _mm256_setzero_ps());
        }

        for (size_t j = (layer.outSize / 8) * 8; j < layer.outSize; j++) {
            layer.weights(j, i) -= learningRate * layer.weightsGradients(j, i) / static_cast<DTYPE>(batchSize);
            layer.weightsGradients(j, i) = 0;
        }
    }
}
#endif

void applyGradientsOnHost(Layer& layer, size_t batchSize, DTYPE learningRate) {
#if defined __AVX2__ || defined __AVX__
    applyBiasGradients(layer, batchSize, learningRate);

    applyWeightGradients(layer, batchSize, learningRate);
#else
    for (size_t i = 0; i < layer.outSize; i++) {
        layer.biases[i] -= learningRate * layer.biasesGradients[i] / static_cast<DTYPE>(batchSize);
        layer.biasesGradients[i] = 0;
        for (size_t j = 0; j < layer.inSize; j++) {
            layer.weights(j, i) -= learningRate * layer.weightsGradients(j, i) / static_cast<DTYPE>(batchSize);
            layer.weightsGradients(j, i) = 0;
        }
    }
#endif
}
