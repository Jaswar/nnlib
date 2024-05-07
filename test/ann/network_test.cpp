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
    Tensor y = Tensor::construct2d({{1}, {0}, {0}, {1}});

    Network network = Network(2, false);
    network.add(5, "relu");
    network.add(1, "sigmoid");

    std::vector<Metric*> metrics = {new MeanSquaredError()};
    network.train(X, y, 500, 1, 0.1, new MeanSquaredError(), metrics);

}
//
//TEST(network, test2) {
//    Layer l1 = Layer(2, 1, "linear", HOST);
//
//    sTensor X = std::make_shared<Tensor>(Tensor::construct2d({{0, 0}, {0, 1}, {1, 0}, {1, 1}}));
//    sTensor y = std::make_shared<Tensor>(Tensor::construct2d({{1}, {0}, {0}, {1}}));
//
////    sTensor w = std::make_shared<Tensor>(Tensor::construct2d({{0.1}, {0.2}}));
////    sTensor b = std::make_shared<Tensor>(Tensor::construct1d({0.3}));
//
//    MeanSquaredError lossFn = MeanSquaredError();
//    for (int i = 0; i < 5000000; i++) {
//        l1.biases->useGrad();
//        l1.weights->useGrad();
//
//        sTensor output = l1.forward(X);
//
//        sTensor loss = hadamard(subtract(output, y), subtract(output, y));
//        loss = sum(loss);
//        loss->backward();
//
//        l1.applyGradients(4, 0.1);
//    }
//
//}
//
//TEST(network, test3) {
//    sTensor a = std::make_shared<Tensor>(Tensor::construct2d({{0, 0}, {0, 1}, {1, 0}, {1, 1}}));
//    sTensor b = std::make_shared<Tensor>(Tensor::construct2d({{0, 0}, {0, 1}, {1, 0}, {1, 1}}));
//
//
//    for (int i = 0; i < 10000000; i++) {
//        a->useGrad();
//        b->useGrad();
//
//        sTensor r = hadamard(a, a);
//        sTensor loss = sum(r);
//        loss->backward();
//    }
//
//}
