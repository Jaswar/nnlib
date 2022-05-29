//
// Created by Jan Warchocki on 10/03/2022.
//

#include <utility>
#include "../../include/vector.h"
#include "../../include/matrix.h"
#include "../../include/layer.h"

#ifndef NNLIB_BACKPROPAGATION_CUH
#define NNLIB_BACKPROPAGATION_CUH

void applyGradients(Layer& layer, size_t batchSize, DTYPE learningRate = 0.01);

void applyGradientsOnHost(Layer& layer, size_t batchSize, DTYPE learningRate = 0.01);
void applyGradientsOnDevice(Layer& layer, size_t batchSize, DTYPE learningRate = 0.01);

#endif //NNLIB_BACKPROPAGATION_CUH
