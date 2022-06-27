//
// Created by Jan Warchocki on 27/06/2022.
//

#include <gtest/gtest.h>
#include <vector.h>
#include "../utils.h"
#include "../assertions.h"

TEST(vectorOperationsHost, add) {
    const Vector& v1 = constructVector({1, 2});
    const Vector& v2 = constructVector({2, 2});
    Vector result = Vector(2);

    add(v1, v2, result);

    ASSERT_VECTOR_EQ(result, {3, 4});
}

TEST(vectorOperationsHost, subtract) {
    const Vector& v1 = constructVector({1, 2});
    const Vector& v2 = constructVector({2, 2});
    Vector result = Vector(2);

    subtract(v1, v2, result);

    ASSERT_VECTOR_EQ(result, {-1, 0});
}

TEST(vectorOperationsHost, multiplyConstantVector) {
    const Vector& v = constructVector({-2, 2});
    Vector result = Vector(2);

    multiply(v, 4, result);

    ASSERT_VECTOR_EQ(result, {-8, 8});
}

TEST(vectorOperationsHost, multiplyVectorConstant) {
    const Vector& v = constructVector({-2, 2});
    Vector result = Vector(2);

    multiply(4, v, result);

    ASSERT_VECTOR_EQ(result, {-8, 8});
}