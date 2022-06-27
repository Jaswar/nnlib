//
// Created by Jan Warchocki on 27/06/2022.
//

#ifndef NNLIB_ASSERTIONS_H
#define NNLIB_ASSERTIONS_H

#include <gtest/gtest.h>
#include <vector.h>
#include <matrix.h>

#define ASSERT_VECTOR(result, ...) ASSERT_TRUE(assertEqual(result, __VA_ARGS__))
::testing::AssertionResult assertEqual(const Vector& result, std::initializer_list<DTYPE> expected);

#define ASSERT_MATRIX(result, ...) ASSERT_TRUE(assertEqual(result, __VA_ARGS__))
::testing::AssertionResult assertEqual(const Matrix& result, std::initializer_list<std::initializer_list<DTYPE>> expected);

#endif //NNLIB_ASSERTIONS_H
