//
// Created by Jan Warchocki on 27/06/2022.
//

#include <gtest/gtest.h>
#include <vector.h>
#include "../utils.h"
#include "../assertions.h"

TEST(vectorOperationsHost, add) {
    const Vector& v1 = constructVector({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    const Vector& v2 = constructVector({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    Vector result = Vector(10);

    add(v1, v2, result);

    ASSERT_VECTOR_EQ(result, {2, 4, 6, 8, 10, 12, 14, 16, 18, 20});
}

TEST(vectorOperationsHost, subtract) {
    const Vector& v1 = constructVector({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    const Vector& v2 = constructVector({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    Vector result = Vector(10);

    subtract(v1, v2, result);

    ASSERT_VECTOR_EQ(result, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
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