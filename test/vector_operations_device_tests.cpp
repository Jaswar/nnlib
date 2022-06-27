//
// Created by Jan Warchocki on 27/06/2022.
//

#include <gtest/gtest.h>
#include <vector.h>
#include <verify.cuh>
#include "utils.h"

#ifdef HAS_CUDA

TEST(vector_operations_device, add) {
    Vector v1 = constructVector({1, 2});
    Vector v2 = constructVector({2, 2});
    Vector result = Vector(2, DEVICE);

    v1.moveToDevice();
    v2.moveToDevice();

    add(v1, v2, result);

    result.moveToHost();

    assertEqual(result, {3, 4});
}

TEST(vector_operations_device, subtract) {
    Vector v1 = constructVector({1, 2});
    Vector v2 = constructVector({2, 2});
    Vector result = Vector(2, DEVICE);

    v1.moveToDevice();
    v2.moveToDevice();

    subtract(v1, v2, result);

    result.moveToHost();

    assertEqual(result, {-1, 0});
}

TEST(vector_operations_device, multiply_constant_vector) {
    Vector v = constructVector({-2, 2});
    Vector result = Vector(2, DEVICE);

    v.moveToDevice();

    multiply(v, 4, result);

    result.moveToHost();

    assertEqual(result, {-8, 8});
}

TEST(vector_operations_device, multiply_vector_constant) {
    Vector v = constructVector({-2, 2});
    Vector result = Vector(2, DEVICE);

    v.moveToDevice();

    multiply(4, v, result);

    result.moveToHost();

    assertEqual(result, {-8, 8});
}

#endif