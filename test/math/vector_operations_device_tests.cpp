//
// Created by Jan Warchocki on 27/06/2022.
//

#include <gtest/gtest.h>
#include <vector.h>
#include <verify.cuh>
#include "../utils.h"
#include "../assertions.h"

#ifdef HAS_CUDA

TEST(vector_operations_device, add) {
    Vector v1 = constructVector({1, 2}, DEVICE);
    Vector v2 = constructVector({2, 2}, DEVICE);
    Vector result = Vector(2, DEVICE);

    add(v1, v2, result);

    result.moveToHost();

    ASSERT_VECTOR(result, {3, 4});
}

TEST(vector_operations_device, subtract) {
    Vector v1 = constructVector({1, 2}, DEVICE);
    Vector v2 = constructVector({2, 2}, DEVICE);
    Vector result = Vector(2, DEVICE);

    subtract(v1, v2, result);

    result.moveToHost();

    ASSERT_VECTOR(result, {-1, 0});
}

TEST(vector_operations_device, multiply_constant_vector) {
    Vector v = constructVector({-2, 2}, DEVICE);
    Vector result = Vector(2, DEVICE);

    multiply(v, 4, result);

    result.moveToHost();

    ASSERT_VECTOR(result, {-8, 8});
}

TEST(vector_operations_device, multiply_vector_constant) {
    Vector v = constructVector({-2, 2}, DEVICE);
    Vector result = Vector(2, DEVICE);

    multiply(4, v, result);

    result.moveToHost();

    ASSERT_VECTOR(result, {-8, 8});
}

#endif