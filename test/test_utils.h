/**
 * @file test_utils.h
 * @brief 
 * @author Jan Warchocki
 * @date 13 October 2022
 */

#ifndef NNLIB_TEST_UTILS_H
#define NNLIB_TEST_UTILS_H

#include <tensor.h>

float getRandomValue();

template<typename... Args>
Tensor initializeRandom(Args... args) {
    Tensor tensor = Tensor(args...);

    for (size_t i = 0; i < tensor.size; i++) {
        tensor.host[i] = getRandomValue();
    }

    return tensor;
}

#endif //NNLIB_TEST_UTILS_H
