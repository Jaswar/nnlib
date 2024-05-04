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
    sTensor a = std::make_shared<Tensor>(Tensor::construct2d({{1, 0}, {0, 1}}));
    sTensor b = std::make_shared<Tensor>(Tensor::construct2d({{3, 4}, {6, 7}}));
    a->useGrad();
    b->useGrad();

    Matmul matmul;
    sTensor result = matmul.forward({a, b});

    Matmul matmul2;
    sTensor result2 = matmul2.forward({a, result});
    result2->backward();
    std::cout << *a->grad << std::endl;
    std::cout << *b->grad << std::endl;
}
