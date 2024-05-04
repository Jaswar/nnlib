/**
 * @file function.cpp
 * @brief
 *
 * @author Jan Warchocki
 * @date 03 May 2024
 *
 */

#include "functions.h"

sTensor Matmul::forwardFn(const std::vector<sTensor>& args) {
    cacheA = std::make_shared<Tensor>(*args[0]);
    cacheB = std::make_shared<Tensor>(*args[1]);
    sTensor result = std::make_shared<Tensor>(multiply(*args[0], *args[1]));
    return result;
}

std::vector<sTensor> Matmul::backwardFn(sTensor grad) {
    sTensor gradA = std::make_shared<Tensor>(multiply(*grad, transpose(*cacheB)));
    sTensor gradB = std::make_shared<Tensor>(multiply(transpose(*cacheA), *grad));
    return {gradA, gradB};
}