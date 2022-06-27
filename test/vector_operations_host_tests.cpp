//
// Created by Jan Warchocki on 27/06/2022.
//

#include <gtest/gtest.h>
#include <vector.h>
#include "utils.h"

TEST(vector_operations_host, add) {
    const Vector& v1 = constructVector({1, 2});
    const Vector& v2 = constructVector({2, 2});
    Vector result = Vector(2, HOST);

    add(v1, v2, result);

    ASSERT_EQ(result[0], 3);
    ASSERT_EQ(result[1], 4);
}

TEST(vector_operations_host, subtract) {
    const Vector& v1 = constructVector({1, 2});
    const Vector& v2 = constructVector({2, 2});
    Vector result = Vector(2, HOST);

    subtract(v1, v2, result);

    ASSERT_EQ(result[0], -1);
    ASSERT_EQ(result[1], 0);
}

TEST(vector_operations_host, multiply_constant_vector) {
    const Vector& v = constructVector({-2, 2});
    Vector result = Vector(2, HOST);

    multiply(v, 4, result);

    ASSERT_EQ(result[0], -8);
    ASSERT_EQ(result[1], 8);
}

TEST(vector_operations_host, multiply_vector_constant) {
    const Vector& v = constructVector({-2, 2});
    Vector result = Vector(2, HOST);

    multiply(4, v, result);

    ASSERT_EQ(result[0], -8);
    ASSERT_EQ(result[1], 8);
}