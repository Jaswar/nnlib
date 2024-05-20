//
// Created by Jan Warchocki on 27/06/2022.
//

#ifndef NNLIB_ASSERTIONS_H
#define NNLIB_ASSERTIONS_H

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <tensor.h>

#define ASSERT_TENSOR_EQ(result, ...) ASSERT_TRUE(assertEqual(result, __VA_ARGS__))
#define RC_ASSERT_TENSOR_EQ(result, ...) RC_ASSERT(assertEqual(result, __VA_ARGS__) == ::testing::AssertionSuccess())
template<typename T>
::testing::AssertionResult assertEqual(const Tensor<T>& result, const Tensor<T>& expected) {
    if (result.shape != expected.shape) {
        return ::testing::AssertionFailure() << "The shapes of the tensors are different.";
    }

    for (size_t i = 0; i < result.size; i++) {
        if (result.data[i] != expected.data[i]) {
            return ::testing::AssertionFailure()
                   << "The tensors are different at index " << i << " (expected: " << expected.data[i] << " was "
                   << result.data[i] << ")";
        }
    }
    return ::testing::AssertionSuccess();
}

#define ASSERT_TENSOR_EQ_1D(result, ...) ASSERT_TRUE(assertEqual1d(result, __VA_ARGS__))
#define RC_ASSERT_TENSOR_EQ_1D(result, ...) \
    RC_ASSERT(assertEqual1d(result, __VA_ARGS__) == ::testing::AssertionSuccess())
template<typename T>
::testing::AssertionResult assertEqual1d(const Tensor<T>& result, const std::vector<T>& expected) {
    Tensor exp = Tensor<T>::construct1d(expected);
    return assertEqual(result, exp);
}

#define ASSERT_TENSOR_EQ_2D(result, ...) ASSERT_TRUE(assertEqual2d(result, __VA_ARGS__))
#define RC_ASSERT_TENSOR_EQ_2D(result, ...) \
    RC_ASSERT(assertEqual2d(result, __VA_ARGS__) == ::testing::AssertionSuccess())
template<typename T>
::testing::AssertionResult assertEqual2d(const Tensor<T>& result, const std::vector<std::vector<T>>& expected) {
    Tensor exp = Tensor<T>::construct2d(expected);
    return assertEqual(result, exp);
}

bool withinBounds(float v1, float v2, float delta, bool relative);

#define ASSERT_TENSOR_CLOSE(result, ...) ASSERT_TRUE(assertClose(result, __VA_ARGS__))
#define RC_ASSERT_TENSOR_CLOSE(result, ...) RC_ASSERT(assertClose(result, __VA_ARGS__) == ::testing::AssertionSuccess())
template<typename T>
::testing::AssertionResult assertClose(const Tensor<T>& result, const Tensor<T>& expected, float delta = 5e-5,
                                       bool relative = false) {
    if (result.shape != expected.shape) {
        return ::testing::AssertionFailure() << "The shapes of the tensors are different.";
    }

    for (size_t i = 0; i < result.size; i++) {
        if (!withinBounds(result.data[i], expected.data[i], delta, relative)) {
            return ::testing::AssertionFailure()
                   << "The tensors are different at index " << i << " (expected: " << expected.data[i] << " was "
                   << result.data[i] << ")";
        }
    }
    return ::testing::AssertionSuccess();
}

#define ASSERT_TENSOR_CLOSE_1D(result, ...) ASSERT_TRUE(assertClose1d(result, __VA_ARGS__))
#define RC_ASSERT_TENSOR_CLOSE_1D(result, ...) \
    RC_ASSERT(assertClose1d(result, __VA_ARGS__) == ::testing::AssertionSuccess())
template<typename T>
::testing::AssertionResult assertClose1d(const Tensor<T>& result, const std::vector<T>& expected, float delta = 5e-5,
                                         bool relative = false) {
    std::shared_ptr<Tensor<T>> exp = Tensor<T>::construct1d(expected);
    return assertClose(result, *exp, delta, relative);
}

#define ASSERT_TENSOR_CLOSE_2D(result, ...) ASSERT_TRUE(assertClose2d(result, __VA_ARGS__))
#define RC_ASSERT_TENSOR_CLOSE_2D(result, ...) \
    RC_ASSERT(assertClose2d(result, __VA_ARGS__) == ::testing::AssertionSuccess())
template<typename T>
::testing::AssertionResult assertClose2d(const Tensor<T>& result, const std::vector<std::vector<T>>& expected,
                                         float delta = 5e-5, bool relative = false) {
    std::shared_ptr<Tensor<T>> exp = Tensor<T>::construct2d(expected);
    return assertClose(result, *exp, delta, relative);
}

#endif //NNLIB_ASSERTIONS_H
