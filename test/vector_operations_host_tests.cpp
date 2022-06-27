//
// Created by Jan Warchocki on 27/06/2022.
//

#include <gtest/gtest.h>
#include <vector.h>
#include "utils.h"

TEST(vector_operations_host, add) {
    const Vector& v1 = constructVector({1, 2});
    const Vector& v2 = constructVector({2, 2});
    Vector result = Vector(2);

    add(v1, v2, result);

    assertEqual(result, {3, 4});
}

TEST(vector_operations_host, subtract) {
    const Vector& v1 = constructVector({1, 2});
    const Vector& v2 = constructVector({2, 2});
    Vector result = Vector(2);

    subtract(v1, v2, result);

    assertEqual(result, {-1, 0});
}

TEST(vector_operations_host, multiply_constant_vector) {
    const Vector& v = constructVector({-2, 2});
    Vector result = Vector(2);

    multiply(v, 4, result);

    assertEqual(result, {-8, 8});
}

TEST(vector_operations_host, multiply_vector_constant) {
    const Vector& v = constructVector({-2, 2});
    Vector result = Vector(2);

    multiply(4, v, result);

    assertEqual(result, {-8, 8});
}