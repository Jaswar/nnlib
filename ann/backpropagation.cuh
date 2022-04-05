//
// Created by Jan Warchocki on 10/03/2022.
//

#include <utility>
#include "../math/vector.h"
#include "../math/matrix.h"
#include "layer.h"

#ifndef NNLIB_BACKPROPAGATION_CUH
#define NNLIB_BACKPROPAGATION_CUH

void computeGradients(Layer& layer, const Matrix& delta, const Matrix& previousWeights,
                      size_t batchSize, bool isLastLayer);

void applyGradients(Layer& layer, size_t batchSize, DTYPE learningRate = 0.01);

#endif //NNLIB_BACKPROPAGATION_CUH
