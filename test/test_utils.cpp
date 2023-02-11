/**
 * @file test_utils.cpp
 * @brief
 * @author Jan Warchocki
 * @date 13 October 2022
 */

#include "test_utils.h"

float getRandomValue() {
    return ((static_cast<float>(rand()) / RAND_MAX) * 2 - 1);
}

std::vector<float> rcFloatVectorInRange(size_t size, float lower, float upper, size_t precision) {
    auto fPrecision = static_cast<float>(precision);
    int lowerInt = static_cast<int>(lower * fPrecision);
    int upperInt = static_cast<int>(upper * fPrecision);

    const auto resultInt = *NO_SHRINK(rc::gen::container<std::vector<int>>(size, rc::gen::inRange<int>(lowerInt, upperInt)));
    std::vector<float> result = std::vector<float>(size);
    std::transform(resultInt.begin(), resultInt.end(), result.begin(), [&fPrecision](int x) {
        return static_cast<float>(x) / fPrecision;
    });

    return result;
}