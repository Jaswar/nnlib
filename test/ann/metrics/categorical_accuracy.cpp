/**
 * @file categorical_accuracy.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 06 May 2023
 */

#include <gtest/gtest.h>
#include <cmath>
#include <tensor.h>
#include <metric.h>

TEST(categorical_accuracy, calculate_metric) {
    Tensor expected = Tensor::construct2d({{0, 0, 1}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}});
    Tensor actual = Tensor::construct2d({{1.1f, 0, 0}, {0.25f, 0, 0}, {0, 0, 1.5f}, {0, 0.77f, 0}});

    CategoricalAccuracy metric = CategoricalAccuracy();
    float accuracy = metric.calculateMetric(expected, actual);
    ASSERT_EQ(accuracy, 0.25f);
}
