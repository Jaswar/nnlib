//
// Created by Jan Warchocki on 27/06/2022.
//

#include <gtest/gtest.h>
#include <vector.h>

TEST(vector_operations_host, add) {
    Vector v1 = Vector(2, HOST);
    Vector v2 = Vector(2, HOST);
    Vector result = Vector(2, HOST);

    v1[0] = 1;
    v1[1] = 2;
    v2[0] = 2;
    v2[1] = 2;

    add(v1, v2, result);

    ASSERT_EQ(result[0], 3);
    ASSERT_EQ(result[1], 4);
}

TEST(vector_operations_host, subtract) {
    Vector v1 = Vector(2, HOST);
    Vector v2 = Vector(2, HOST);
    Vector result = Vector(2, HOST);

    v1[0] = 1;
    v1[1] = 2;
    v2[0] = 2;
    v2[1] = 2;

    subtract(v1, v2, result);

    ASSERT_EQ(result[0], -1);
    ASSERT_EQ(result[1], 0);
}

TEST(vector_operations_host, multiply_constant_vector) {
    Vector v = Vector(2, HOST);
    Vector result = Vector(2, HOST);

    v[0] = -2;
    v[1] = 2;

    multiply(v, 4, result);

    ASSERT_EQ(result[0], -8);
    ASSERT_EQ(result[1], 8);
}

TEST(vector_operations_host, multiply_vector_constant) {
    Vector v = Vector(2, HOST);
    Vector result = Vector(2, HOST);

    v[0] = -2;
    v[1] = 2;

    multiply(4, v, result);

    ASSERT_EQ(result[0], -8);
    ASSERT_EQ(result[1], 8);
}