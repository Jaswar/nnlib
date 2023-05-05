/**
 * @file network_test.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 10 March 2023
 */

#include "network.h"
#include <gtest/gtest.h>

TEST(network, test) {
    Tensor X = Tensor::construct2d({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
    Tensor y = Tensor::construct2d({{0}, {0}, {0}, {1}});

    Network network = Network(2, true);
    network.add(5, "relu");
    network.add(1, "sigmoid");

    std::vector<Metric*> metrics = {new MeanSquaredError()};
    network.train(X, y, 500, 1, 0.1, new MeanSquaredError(), metrics);

}
