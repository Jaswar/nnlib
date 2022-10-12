//
// Created by Jan Warchocki on 27/06/2022.
//

#ifndef NNLIB_ASSERTIONS_H
#define NNLIB_ASSERTIONS_H

#include <gtest/gtest.h>
#include <tensor.h>

#define ASSERT_EQ_1D(result, ...) ASSERT_TRUE(assertEqual(result, __VA_ARGS__))
::testing::AssertionResult assertEqual(const Tensor& result, std::initializer_list<DTYPE> expected);

#define ASSERT_EQ_2D(result, ...) ASSERT_TRUE(assertEqual(result, __VA_ARGS__))
::testing::AssertionResult assertEqual(const Tensor& result,
                                       std::initializer_list<std::initializer_list<DTYPE>> expected);

#define ASSERT_CLOSE_1D(result, ...) ASSERT_TRUE(assertClose(result, __VA_ARGS__))
::testing::AssertionResult assertClose(const Tensor& result, std::initializer_list<DTYPE> expected, float delta = 5e-5);

#define ASSERT_CLOSE_2D(result, ...) ASSERT_TRUE(assertClose(result, __VA_ARGS__))
::testing::AssertionResult
assertClose(const Tensor& result, std::initializer_list<std::initializer_list<DTYPE>> expected, float delta = 5e-5);

#endif //NNLIB_ASSERTIONS_H
