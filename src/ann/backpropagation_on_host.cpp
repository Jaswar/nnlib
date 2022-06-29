//
// Created by Jan Warchocki on 29/05/2022.
//

#include <layer.h>

void applyGradientsOnHost(Layer& layer, size_t batchSize, DTYPE learningRate) {
    for (size_t i = 0; i < layer.outSize; i++) {
        layer.biases[i] -= learningRate * layer.biasesGradients[i] / static_cast<DTYPE>(batchSize);
        layer.biasesGradients[i] = 0;
        for (size_t j = 0; j < layer.inSize; j++) {
            layer.weights(j, i) -= learningRate * layer.weightsGradients(j, i) / static_cast<DTYPE>(batchSize);
            layer.weightsGradients(j, i) = 0;
        }
    }
}
