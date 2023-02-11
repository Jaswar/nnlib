/**
 * @file test_utils.h
 * @brief 
 * @author Jan Warchocki
 * @date 13 October 2022
 */

#ifndef NNLIB_TEST_UTILS_H
#define NNLIB_TEST_UTILS_H

#include <tensor.h>
#include <rapidcheck.h>

#define NO_SHRINK(...) rc::gen::noShrink(__VA_ARGS__)

float getRandomValue();

std::vector<float> rcFloatVectorInRange(size_t size, float lower, float upper, size_t precision = 1e6);

template<typename... Args>
Tensor initializeRandom(Args... args) {
    Tensor tensor = Tensor(args...);

    for (size_t i = 0; i < tensor.size; i++) {
        tensor.data[i] = getRandomValue();
    }

    return tensor;
}

#endif //NNLIB_TEST_UTILS_H
