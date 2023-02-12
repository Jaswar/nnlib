/**
 * @file binary_crossentropy.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 12 February 2023
 */

#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include <tensor.h>
#include <loss.h>
#include "../../test_utils.h"
#include "../../assertions.h"
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#define DELTA 0.0005

RC_GTEST_PROP(binary_cross_entropy, calculate_loss, ()) {
    const auto numSamples = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e5));
    const rc::Gen<float> allowed = rc::gen::element(0.0f, 1.0f);
    const auto dataTargets = *NO_SHRINK(rc::gen::container<std::vector<float>>(numSamples, allowed));
    const auto dataPredictions = rcFloatVectorInRange(numSamples, 1e-4, 1 - 1e-4);

    Tensor targets = Tensor(numSamples, 1);
    std::copy(dataTargets.begin(), dataTargets.end(), targets.data);
    Tensor predictions = Tensor(numSamples, 1);
    std::copy(dataPredictions.begin(), dataPredictions.end(), predictions.data);

    float expected = 0;
    for (size_t sample = 0; sample < numSamples; sample++) {
        float y = targets.data[sample];
        float yHat = predictions.data[sample];
        expected += y * logf(yHat) + (1 - y) * logf(1 - yHat);
    }
    expected /= -static_cast<float>(numSamples);

    BinaryCrossEntropy error = BinaryCrossEntropy();
    float result = error.calculateLoss(targets, predictions);

    RC_ASSERT(std::abs(result - expected) <= DELTA);
}

RC_GTEST_PROP(binary_cross_entropy, calculate_derivatives, ()) {
    const auto numSamples = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e5));
    const rc::Gen<float> allowed = rc::gen::element(0.0f, 1.0f);
    const auto dataTargets = *NO_SHRINK(rc::gen::container<std::vector<float>>(numSamples, allowed));
    const auto dataPredictions = rcFloatVectorInRange(numSamples, 1e-4, 1 - 1e-4);

    Tensor targets = Tensor(numSamples, 1);
    std::copy(dataTargets.begin(), dataTargets.end(), targets.data);
    Tensor predictions = Tensor(numSamples, 1);
    std::copy(dataPredictions.begin(), dataPredictions.end(), predictions.data);
    Tensor expected = Tensor(numSamples, 1);
    Tensor result = Tensor(numSamples, 1);

    for (size_t sample = 0; sample < numSamples; sample++) {
        float y = targets.data[sample];
        float yHat = predictions.data[sample];
        expected.data[sample] = (yHat - y) / (yHat * (1 - yHat));
    }

    BinaryCrossEntropy error = BinaryCrossEntropy();
    error.calculateDerivatives(targets, predictions, result);

    RC_ASSERT_TENSOR_EQ(result, expected);
}
