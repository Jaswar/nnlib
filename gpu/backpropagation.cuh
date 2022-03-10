//
// Created by Jan Warchocki on 10/03/2022.
//

#include <utility>
#include "../math/vector.h"
#include "../math/matrix.h"
#include "../ann/layer.h"

#ifndef NNLIB_BACKPROPAGATION_CUH
#define NNLIB_BACKPROPAGATION_CUH

Vector backpropagation(Layer& layer, const Vector& delta, const Matrix& previousWeights,
                bool isLastLayer, DTYPE learningRate);

#endif //NNLIB_BACKPROPAGATION_CUH
