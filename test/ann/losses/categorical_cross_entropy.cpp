/**
 * @file categorical_cross_entropy.cpp
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

#define DELTA 0.005

RC_GTEST_PROP(categorical_cross_entropy, calculate_loss, ()) {
    const auto numSamples = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e4));
    const auto numClasses = *NO_SHRINK(rc::gen::inRange<size_t>(1, 100));
    const rc::Gen<float> allowed = rc::gen::element(0.0f, 1.0f);
    const auto dataTargets = *NO_SHRINK(rc::gen::container<std::vector<float>>(numSamples * numClasses, allowed));
    const auto dataPredictions = rcFloatVectorInRange(numSamples * numClasses, 1e-4, 1 - 1e-4);

    Tensor targets = Tensor(numSamples, numClasses);
    std::copy(dataTargets.begin(), dataTargets.end(), targets.data);
    Tensor predictions = Tensor(numSamples, numClasses);
    std::copy(dataPredictions.begin(), dataPredictions.end(), predictions.data);

    float expected = 0;
    for (size_t sample = 0; sample < numSamples; sample++) {
        float sum = 0;
        for (size_t cls = 0; cls < numClasses; cls++) {
            sum += predictions(sample, cls);
        }

        for (size_t cls = 0; cls < numClasses; cls++) {
            float y = targets(sample, cls);
            float yHat = predictions(sample, cls) / sum;
            expected += y * logf(yHat);
        }
    }
    expected /= -static_cast<float>(numSamples);

    CategoricalCrossEntropy error = CategoricalCrossEntropy();
    float result = error.calculateLoss(targets, predictions);

    RC_ASSERT(std::abs(result - expected) <= DELTA);
}

RC_GTEST_PROP(categorical_cross_entropy, calculate_derivatives, ()) {
    const auto numSamples = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e4));
    const auto numClasses = *NO_SHRINK(rc::gen::inRange<size_t>(1, 100));
    const rc::Gen<float> allowed = rc::gen::element(0.0f, 1.0f);
    const auto dataTargets = *NO_SHRINK(rc::gen::container<std::vector<float>>(numSamples * numClasses, allowed));
    const auto dataPredictions = rcFloatVectorInRange(numSamples * numClasses, 1e-4, 1 - 1e-4);

    Tensor targets = Tensor(numSamples, numClasses);
    std::copy(dataTargets.begin(), dataTargets.end(), targets.data);
    Tensor predictions = Tensor(numSamples, numClasses);
    std::copy(dataPredictions.begin(), dataPredictions.end(), predictions.data);
    Tensor result = Tensor(numSamples, numClasses);
    Tensor expected = Tensor(numSamples, numClasses);

    for (size_t sample = 0; sample < numSamples; sample++) {
        float sum = 0;
        for (size_t cls = 0; cls < numClasses; cls++) {
            sum += predictions(sample, cls);
        }

        for (size_t cls = 0; cls < numClasses; cls++) {
            float y = targets(sample, cls);
            float yHat = predictions(sample, cls);
            expected(sample, cls) = -y / yHat + 1 / sum;
        }
    }

    CategoricalCrossEntropy error = CategoricalCrossEntropy();
    error.calculateDerivatives(targets, predictions, result);

    RC_ASSERT_TENSOR_CLOSE(result, expected);
}
