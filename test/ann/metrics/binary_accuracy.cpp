/**
 * @file binary_accuracy.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 07 May 2023
 */

#include <gtest/gtest.h>
#include <cmath>
#include <tensor.h>
#include <metric.h>

TEST(binary_accuracy, calculate_metric) {
    Tensor expected = Tensor::construct2d({{1}, {1}, {1}, {0}, {0}});
    Tensor actual = Tensor::construct2d({{0.9}, {0.5}, {0.25}, {0.1}, {0.5}});

    BinaryAccuracy metric = BinaryAccuracy();
    float accuracy = metric.calculateMetric(expected, actual);
    ASSERT_EQ(accuracy, 0.6f);
}
