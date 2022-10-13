//
// Created by Jan Warchocki on 27/06/2022.
//

#ifndef NNLIB_ASSERTIONS_H
#define NNLIB_ASSERTIONS_H

#include <gtest/gtest.h>
#include <tensor.h>

#define ASSERT_TENSOR_EQ(result, ...) ASSERT_TRUE(assertEqual(result, __VA_ARGS__))
::testing::AssertionResult assertEqual(const Tensor& result, const Tensor& expected);
#define ASSERT_TENSOR_EQ_1D(result, ...) ASSERT_TRUE(assertEqual1d(result, __VA_ARGS__))
::testing::AssertionResult assertEqual1d(const Tensor& result, std::vector<float> expected);
#define ASSERT_TENSOR_EQ_2D(result, ...) ASSERT_TRUE(assertEqual2d(result, __VA_ARGS__))
::testing::AssertionResult assertEqual2d(const Tensor& result, std::vector<std::vector<float>> expected);

#define ASSERT_TENSOR_CLOSE(result, ...) ASSERT_TRUE(assertClose(result, __VA_ARGS__))
::testing::AssertionResult assertClose(const Tensor& result, const Tensor& expected, float delta = 5e-5);
#define ASSERT_TENSOR_CLOSE_1D(result, ...) ASSERT_TRUE(assertClose1d(result, __VA_ARGS__))
::testing::AssertionResult assertClose1d(const Tensor& result, std::vector<float> expected, float delta = 5e-5);
#define ASSERT_TENSOR_CLOSE_2D(result, ...) ASSERT_TRUE(assertClose2d(result, __VA_ARGS__))
::testing::AssertionResult
assertClose2d(const Tensor& result, std::vector<std::vector<float>> expected, float delta = 5e-5);

#endif //NNLIB_ASSERTIONS_H
