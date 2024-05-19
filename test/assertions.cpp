//
// Created by Jan Warchocki on 27/06/2022.
//

#include "assertions.h"

#include <utility>

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

template<typename T>
::testing::AssertionResult assertEqual1d(const Tensor<T>& result, const std::vector<float>& expected) {
    Tensor exp = Tensor<T>::construct1d(expected);
    return assertEqual(result, exp);
}

template<typename T>
::testing::AssertionResult assertEqual2d(const Tensor<T>& result, const std::vector<std::vector<float>>& expected) {
    Tensor exp = Tensor<T>::construct2d(expected);
    return assertEqual(result, exp);
}

bool withinBoundsAbsolute(const float v1, const float v2, const float delta) {
    return std::abs(v1 - v2) <= delta;
}

bool withinBoundsRelative(const float v1, const float v2, const float delta) {
    if (v2 == 0)
        return v1 == 0;
    return std::abs(v1 / v2 - 1) <= delta;
}

bool withinBounds(const float v1, const float v2, const float delta, bool relative) {
    if (relative) {
        return withinBoundsRelative(v1, v2, delta);
    } else {
        return withinBoundsAbsolute(v1, v2, delta);
    }
}

template<typename T>
::testing::AssertionResult assertClose(const Tensor<T>& result, const Tensor<T>& expected, float delta, bool relative) {
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

template<typename T>
::testing::AssertionResult assertClose1d(const Tensor<T>& result, const std::vector<float>& expected, float delta,
                                         bool relative) {
    Tensor exp = Tensor<T>::construct1d(expected);
    return assertClose(result, exp, delta, relative);
}

template<typename T>
::testing::AssertionResult assertClose2d(const Tensor<T>& result, const std::vector<std::vector<float>>& expected,
                                         float delta, bool relative) {
    Tensor exp = Tensor<T>::construct2d(expected);
    return assertClose(result, exp, delta, relative);
}
