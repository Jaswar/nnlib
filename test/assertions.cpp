//
// Created by Jan Warchocki on 27/06/2022.
//

#include "assertions.h"

#include <utility>

::testing::AssertionResult assertEqual(const Tensor& result, const Tensor& expected) {
    if (result.shape != expected.shape) {
        return ::testing::AssertionFailure() << "The shapes of the tensors are different.";
    }

    for (size_t i = 0; i < result.size; i++) {
        if (result.host[i] != expected.host[i]) {
            return ::testing::AssertionFailure()
                   << "The tensors are different at index " << i << " (expected: " << expected.host[i] << " was "
                   << result.host[i] << ")";
        }
    }
    return ::testing::AssertionSuccess();
}

::testing::AssertionResult assertEqual1d(const Tensor& result, const std::vector<float>& expected) {
    Tensor exp = Tensor::construct1d(expected);
    return assertEqual(result, exp);
}

::testing::AssertionResult assertEqual2d(const Tensor& result, const std::vector<std::vector<float>>& expected) {
    Tensor exp = Tensor::construct2d(expected);
    return assertEqual(result, exp);
}

::testing::AssertionResult assertClose(const Tensor& result, const Tensor& expected, float delta) {
    if (result.shape != expected.shape) {
        return ::testing::AssertionFailure() << "The shapes of the tensors are different.";
    }

    for (size_t i = 0; i < result.size; i++) {
        if (std::abs(result.host[i] - expected.host[i]) > delta) {
            return ::testing::AssertionFailure()
                   << "The tensors are different at index " << i << " (expected: " << expected.host[i] << " was "
                   << result.host[i] << ")";
        }
    }
    return ::testing::AssertionSuccess();
}

::testing::AssertionResult assertClose1d(const Tensor& result, const std::vector<float>& expected, float delta) {
    Tensor exp = Tensor::construct1d(expected);
    return assertClose(result, exp, delta);
}

::testing::AssertionResult assertClose2d(const Tensor& result, const std::vector<std::vector<float>>& expected,
                                         float delta) {
    Tensor exp = Tensor::construct2d(expected);
    return assertClose(result, exp, delta);
}
