/**
 * @file mean_squared_error.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 11 February 2023
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

RC_GTEST_PROP(mean_squared_error, calculate_loss, ()) {
    const auto numSamples = *NO_SHRINK(rc::gen::inRange<size_t>(1, 1e4));
    const auto numOutputs = *NO_SHRINK(rc::gen::inRange<size_t>(1, 100));
    const auto size = numSamples * numOutputs;

    std::vector<float> dataTargets = rcFloatVectorInRange(size, -1, 1);
    std::vector<float> dataPredictions = rcFloatVectorInRange(size, -1, 1);

    sTensor targets = std::make_shared<Tensor>(numSamples, numOutputs);
    std::copy(dataTargets.begin(), dataTargets.end(), targets->data);
    sTensor predictions = std::make_shared<Tensor>(numSamples, numOutputs);
    std::copy(dataPredictions.begin(), dataPredictions.end(), predictions->data);

    float expected = 0;
    for (size_t sample = 0; sample < numSamples; sample++) {
        for (size_t output = 0; output < numOutputs; output++) {
            expected += powf((*targets)(sample, output) - (*predictions)(sample, output), 2);
        }
    }
    expected /= static_cast<float>(size);

    MeanSquaredError error = MeanSquaredError();

    float result = error.calculateMetric(targets, predictions);
    // The loss should be the same if we flip the order
    float resultInverted = error.calculateMetric(predictions, targets);

    RC_ASSERT(std::abs(expected - result) <= DELTA);
    RC_ASSERT(std::abs(expected - resultInverted) <= DELTA);
}