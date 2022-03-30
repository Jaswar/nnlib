//
// Created by Jan Warchocki on 10/03/2022.
//

#include <utility>
#include "../math/vector.h"
#include "../math/matrix.h"
#include "layer.h"

#ifndef NNLIB_BACKPROPAGATION_CUH
#define NNLIB_BACKPROPAGATION_CUH

void backpropagation(Layer& layer, const Matrix& delta, const Matrix& previousWeights,
                bool isLastLayer, DTYPE learningRate);

void applyGradient(Layer& layer);

#endif //NNLIB_BACKPROPAGATION_CUH
