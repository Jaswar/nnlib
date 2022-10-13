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
