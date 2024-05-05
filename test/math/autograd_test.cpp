/**
 * @file autograd_test.cpp
 * @brief
 *
 * @author Jan Warchocki
 * @date 04 May 2024
 *
 */

#include "functions.h"
#include "tensor.h"
#include <gtest/gtest.h>

TEST(autograd, test) {
    sTensor a = std::make_shared<Tensor>(Tensor::construct2d({{1, 2}, {2, 5}}));
    sTensor b = std::make_shared<Tensor>(Tensor::construct2d({{3, 4}, {6, 7}}));
    a->move(DEVICE);
    b->move(DEVICE);
    a->useGrad();
    b->useGrad();

    sTensor result = multiply(a, b);

    sTensor result2 = multiply(a, result);
    sTensor loss = sum(result2);
    loss = hadamard(loss, loss);
    loss->backward();

    a->move(HOST);
    b->move(HOST);

    std::cout << *a->grad << std::endl;
    std::cout << *b->grad << std::endl;
}
