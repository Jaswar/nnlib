//
// Created by Jan Warchocki on 10/03/2022.
//

#include "backpropagation.cuh"
#include "verify.cuh"
#include "../gpu/allocation_gpu.cuh"
#include "../gpu/assert.cuh"

void applyGradients(Layer& layer, size_t batchSize, DTYPE learningRate) {
    if (layer.location == HOST) {
        applyGradientsOnHost(layer, batchSize, learningRate);
    } else {
        applyGradientsOnDevice(layer, batchSize, learningRate);
    }
}
