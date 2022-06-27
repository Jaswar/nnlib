//
// Created by Jan Warchocki on 27/06/2022.
//

#include <gtest/gtest.h>
#include <vector.h>
#include <verify.cuh>
#include "../utils.h"
#include "../assertions.h"

#ifdef HAS_CUDA

TEST(vectorOperationsDevice, add) {
    Vector v1 = constructVector({1, 2}, DEVICE);
    Vector v2 = constructVector({2, 2}, DEVICE);
    Vector result = Vector(2, DEVICE);

    add(v1, v2, result);

    result.moveToHost();

    ASSERT_VECTOR_EQ(result, {3, 4});
}

TEST(vectorOperationsDevice, subtract) {
    Vector v1 = constructVector({1, 2}, DEVICE);
    Vector v2 = constructVector({2, 2}, DEVICE);
    Vector result = Vector(2, DEVICE);

    subtract(v1, v2, result);

    result.moveToHost();

    ASSERT_VECTOR_EQ(result, {-1, 0});
}

TEST(vectorOperationsDevice, multiplyConstantVector) {
    Vector v = constructVector({-2, 2}, DEVICE);
    Vector result = Vector(2, DEVICE);

    multiply(v, 4, result);

    result.moveToHost();

    ASSERT_VECTOR_EQ(result, {-8, 8});
}

TEST(vectorOperationsDevice, multiplyVectorConstant) {
    Vector v = constructVector({-2, 2}, DEVICE);
    Vector result = Vector(2, DEVICE);

    multiply(4, v, result);

    result.moveToHost();

    ASSERT_VECTOR_EQ(result, {-8, 8});
}

#endif