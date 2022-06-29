//
// Created by Jan Warchocki on 10/03/2022.
//

#include "backpropagation.h"
#include "../gpu/allocation_gpu.cuh"
#include "../gpu/assert.cuh"
#include "verify.cuh"

void applyGradients(Layer& layer, size_t batchSize, DTYPE learningRate) {
    if (layer.location == HOST) {
        applyGradientsOnHost(layer, batchSize, learningRate);
    } else {
        applyGradientsOnDevice(layer, batchSize, learningRate);
    }
}
