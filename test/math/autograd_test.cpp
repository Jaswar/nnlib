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
#include "../assertions.h"

void testAdd(bool useDevice) {
    sTensor a = std::make_shared<Tensor>(Tensor::construct2d({{1, 2}, {2, 5}}));
    sTensor b = std::make_shared<Tensor>(Tensor::construct2d({{3, 4}, {6, 7}}));
    if (useDevice) {
        a->move(DEVICE);
        b->move(DEVICE);
    }

    a->useGrad();
    b->useGrad();

    sTensor result = add(a, b);
    sTensor loss = sum(result);
    loss->backward();

    a->move(HOST);
    b->move(HOST);

    ASSERT_TENSOR_CLOSE_2D(*a->grad, {{1, 1}, {1, 1}});
    ASSERT_TENSOR_CLOSE_2D(*b->grad, {{1, 1}, {1, 1}});
}

void testAddBroadcast(bool useDevice) {
    sTensor a = std::make_shared<Tensor>(Tensor::construct2d({{1, 2}, {2, 5}, {6, 7}}));
    sTensor b = std::make_shared<Tensor>(Tensor::construct1d({3, 4}));
    if (useDevice) {
        a->move(DEVICE);
        b->move(DEVICE);
    }

    a->useGrad();
    b->useGrad();

    sTensor result = add(a, b);
    sTensor loss = sum(result);
    loss->backward();

    a->move(HOST);
    b->move(HOST);

    ASSERT_TENSOR_CLOSE_2D(*a->grad, {{1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}});
    ASSERT_TENSOR_CLOSE_1D(*b->grad, {3.0, 3.0});
}

void testSubtract(bool useDevice) {
    sTensor a = std::make_shared<Tensor>(Tensor::construct2d({{1, 2}, {2, 5}}));
    sTensor b = std::make_shared<Tensor>(Tensor::construct2d({{3, 4}, {6, 7}}));
    if (useDevice) {
        a->move(DEVICE);
        b->move(DEVICE);
    }

    a->useGrad();
    b->useGrad();

    sTensor result = subtract(a, b);
    sTensor loss = sum(result);
    loss->backward();

    a->move(HOST);
    b->move(HOST);

    ASSERT_TENSOR_CLOSE_2D(*a->grad, {{1, 1}, {1, 1}});
    ASSERT_TENSOR_CLOSE_2D(*b->grad, {{-1, -1}, {-1, -1}});
}

void testHadamard(bool useDevice) {
    sTensor a = std::make_shared<Tensor>(Tensor::construct2d({{1, 2}, {2, 5}}));
    sTensor b = std::make_shared<Tensor>(Tensor::construct2d({{3, 4}, {6, 7}}));
    if (useDevice) {
        a->move(DEVICE);
        b->move(DEVICE);
    }

    a->useGrad();
    b->useGrad();

    sTensor result = hadamard(a, b);
    sTensor loss = sum(result);
    loss->backward();

    a->move(HOST);
    b->move(HOST);

    ASSERT_TENSOR_CLOSE_2D(*a->grad, {{3, 4}, {6, 7}});
    ASSERT_TENSOR_CLOSE_2D(*b->grad, {{1, 2}, {2, 5}});
}

void testDivide(bool useDevice) {
    sTensor a = std::make_shared<Tensor>(Tensor::construct2d({{1, 2}, {2, 5}}));
    sTensor b = std::make_shared<Tensor>(Tensor::construct2d({{3, 4}, {6, 7}}));
    if (useDevice) {
        a->move(DEVICE);
        b->move(DEVICE);
    }

    a->useGrad();
    b->useGrad();

    sTensor result = divide(a, b);
    sTensor loss = sum(result);
    loss->backward();

    a->move(HOST);
    b->move(HOST);

    ASSERT_TENSOR_CLOSE_2D(*a->grad, {{0.3333, 0.2500}, {0.1667, 0.1429}});
    ASSERT_TENSOR_CLOSE_2D(*b->grad, {{-0.1111, -0.1250}, {-0.0556, -0.1020}});
}

void testLog(bool useDevice) {
    sTensor a = std::make_shared<Tensor>(Tensor::construct2d({{1, 2}, {2, 5}}));
    if (useDevice) {
        a->move(DEVICE);
    }

    a->useGrad();

    sTensor result = log(a);
    sTensor loss = sum(result);
    loss->backward();

    a->move(HOST);

    ASSERT_TENSOR_CLOSE_2D(*a->grad, {{1.0, 0.5}, {0.5, 0.2}});
}

void testMultiplyConstant(bool useDevice) {
    sTensor a = std::make_shared<Tensor>(Tensor::construct2d({{1, 2}, {2, 5}}));
    if (useDevice) {
        a->move(DEVICE);
    }

    a->useGrad();

    sTensor result = multiply(a, 2.0f);
    sTensor loss = sum(result);
    loss->backward();

    a->move(HOST);

    ASSERT_TENSOR_CLOSE_2D(*a->grad, {{2.0, 2.0}, {2.0, 2.0}});
}

void testMatVecMul(bool useDevice) {
    sTensor a = std::make_shared<Tensor>(Tensor::construct2d({{1, 2}, {2, 5}}));
    sTensor b = std::make_shared<Tensor>(Tensor::construct1d({3, 4}));
    if (useDevice) {
        a->move(DEVICE);
        b->move(DEVICE);
    }

    a->useGrad();
    b->useGrad();

    sTensor result = multiply(a, b);
    sTensor loss = sum(result);
    loss->backward();

    a->move(HOST);
    b->move(HOST);

    ASSERT_TENSOR_CLOSE_2D(*a->grad, {{3.0, 4.0}, {3.0, 4.0}});
    ASSERT_TENSOR_CLOSE_1D(*b->grad, {3.0, 7.0});
}

void testMatmul(bool useDevice) {
    sTensor a = std::make_shared<Tensor>(Tensor::construct2d({{1, 2}, {2, 5}}));
    sTensor b = std::make_shared<Tensor>(Tensor::construct2d({{3, 4}, {6, 7}}));
    if (useDevice) {
        a->move(DEVICE);
        b->move(DEVICE);
    }

    a->useGrad();
    b->useGrad();

    sTensor result = multiply(a, b);
    sTensor loss = sum(result);
    loss->backward();

    a->move(HOST);
    b->move(HOST);

    ASSERT_TENSOR_CLOSE_2D(*a->grad, {{7.0, 13.0}, {7.0, 13.0}});
    ASSERT_TENSOR_CLOSE_2D(*b->grad, {{3.0, 3.0}, {7.0, 7.0}});
}

void testTranspose(bool useDevice) {
    sTensor a = std::make_shared<Tensor>(Tensor::construct2d({{1, 2}, {2, 5}}));
    if (useDevice) {
        a->move(DEVICE);
    }

    a->useGrad();

    sTensor result = transpose(a);
    sTensor loss = sum(result);
    loss->backward();

    a->move(HOST);

    ASSERT_TENSOR_CLOSE_2D(*a->grad, {{1.0, 1.0}, {1.0, 1.0}});
}

void testRelu(bool useDevice) {
    sTensor a = std::make_shared<Tensor>(Tensor::construct2d({{1, -2}, {-2, 5}}));
    if (useDevice) {
        a->move(DEVICE);
    }

    a->useGrad();

    sTensor result = relu(a);
    sTensor loss = sum(result);
    loss->backward();

    a->move(HOST);

    ASSERT_TENSOR_CLOSE_2D(*a->grad, {{1.0, 0.0}, {0.0, 1.0}});
}

void testSigmoid(bool useDevice) {
    sTensor a = std::make_shared<Tensor>(Tensor::construct2d({{1, -2}, {-2, 0}}));
    if (useDevice) {
        a->move(DEVICE);
    }

    a->useGrad();

    sTensor result = sigmoid(a);
    sTensor loss = sum(result);
    loss->backward();

    a->move(HOST);

    ASSERT_TENSOR_CLOSE_2D(*a->grad, {{0.1966, 0.1050}, {0.1050, 0.2500}});
}

void testCombine(bool useDevice) {
    sTensor a = std::make_shared<Tensor>(Tensor::construct2d({{1, 2}, {2, 5}}));
    sTensor b = std::make_shared<Tensor>(Tensor::construct2d({{3, 4}, {6, 7}}));
    if (useDevice) {
        a->move(DEVICE);
        b->move(DEVICE);
    }

    a->useGrad();
    b->useGrad();

    sTensor result = multiply(a, b);

    sTensor result2 = multiply(a, result);
    sTensor loss = sum(result2);
    loss = hadamard(loss, loss);
    loss->backward();

    a->move(HOST);
    b->move(HOST);

    ASSERT_TENSOR_CLOSE_2D(*a->grad, {{70416, 153872}, {106928, 221680}});
    ASSERT_TENSOR_CLOSE_2D(*b->grad, {{22168, 22168}, {53464, 53464}});
}

void testFork(bool useDevice) {
    sTensor a = std::make_shared<Tensor>(Tensor::construct2d({{1, 2}, {2, 5}}));
    sTensor b = std::make_shared<Tensor>(Tensor::construct2d({{3, 4}, {6, 7}}));
    sTensor c = std::make_shared<Tensor>(Tensor::construct2d({{9, 8}, {11, 10}}));
    if (useDevice) {
        a->move(DEVICE);
        b->move(DEVICE);
        c->move(DEVICE);
    }

    a->useGrad();
    b->useGrad();
    c->useGrad();

    sTensor ab = multiply(a, b);
    sTensor bc = multiply(b, c);
    sTensor result = multiply(ab, bc);
    sTensor loss = sum(result);
    loss->backward();

    a->move(HOST);
    b->move(HOST);
    c->move(HOST);

    ASSERT_TENSOR_CLOSE_2D(*a->grad, {{1401, 2553}, {1401, 2553}});
    ASSERT_TENSOR_CLOSE_2D(*b->grad, {{1272, 1818}, {1982, 3024}});
    ASSERT_TENSOR_CLOSE_2D(*c->grad, {{519, 519}, {631, 631}});
}

void testSimpleNN(bool useDevice) {
    sTensor w1 = std::make_shared<Tensor>(Tensor::construct2d({{1, 2}, {2, 5}, {6, 7}}));
    sTensor b1 = std::make_shared<Tensor>(Tensor::construct1d({3, 4}));
    sTensor w2 = std::make_shared<Tensor>(Tensor::construct2d({{-6, 3}, {-5, 1}}));
    sTensor b2 = std::make_shared<Tensor>(Tensor::construct1d({-3, 4}));
    sTensor x = std::make_shared<Tensor>(Tensor::construct2d({{1, 2, 3}, {4, 6, 6}}));
    if (useDevice) {
        w1->move(DEVICE);
        b1->move(DEVICE);
        w2->move(DEVICE);
        b2->move(DEVICE);
        x->move(DEVICE);
    }

    w1->useGrad();
    b1->useGrad();
    w2->useGrad();
    b2->useGrad();

    sTensor z1 = add(multiply(x, w1), b1);
    sTensor a1 = relu(z1);
    sTensor z2 = add(multiply(a1, w2), b2);
    sTensor a2 = sigmoid(z2);
    sTensor loss = sum(a2);

    loss->backward();

    w1->move(HOST);
    b1->move(HOST);
    w2->move(HOST);
    b2->move(HOST);
    x->move(HOST);

    std::cout << *w1->grad << std::endl;
    std::cout << *b1->grad << std::endl;
    std::cout << *w2->grad << std::endl;
    std::cout << *b2->grad << std::endl;
}

TEST(autograd, test_add_host) {
    testAdd(false);
}

TEST(autograd, test_add_broadcast_host) {
    testAddBroadcast(false);
}

TEST(autograd, test_subtract_host) {
    testSubtract(false);
}

TEST(autograd, test_hadamard_host) {
    testHadamard(false);
}

TEST(autograd, test_divide_host) {
    testDivide(false);
}

TEST(autograd, test_log_host) {
    testLog(false);
}

TEST(autograd, test_multiply_constant_host) {
    testMultiplyConstant(false);
}

TEST(autograd, test_matvecmul_host) {
    testMatVecMul(false);
}

TEST(autograd, test_matmul_host) {
    testMatmul(false);
}

TEST(autograd, test_transpose_host) {
    testTranspose(false);
}

TEST(autograd, test_relu_host) {
    testRelu(false);
}

TEST(autograd, test_sigmoid_host) {
    testSigmoid(false);
}

TEST(autograd, test_combine_host) {
    testCombine(false);
}

TEST(autograd, test_fork_host) {
    testFork(false);
}

TEST(autograd, test_simple_nn_host) {
    testSimpleNN(false);
}

#ifdef __CUDA__

TEST(autograd, test_add_device) {
    testAdd(true);
}

TEST(autograd, test_add_broadcast_device) {
    testAddBroadcast(true);
}

TEST(autograd, test_subtract_device) {
    testSubtract(true);
}

TEST(autograd, test_hadamard_device) {
    testHadamard(true);
}

TEST(autograd, test_divide_device) {
    testDivide(true);
}

TEST(autograd, test_log_device) {
    testLog(true);
}

TEST(autograd, test_multiply_constant_device) {
    testMultiplyConstant(true);
}

TEST(autograd, test_matvecmul_device) {
    testMatVecMul(true);
}

TEST(autograd, test_matmul_device) {
    testMatmul(true);
}

TEST(autograd, test_transpose_device) {
    testTranspose(true);
}

TEST(autograd, test_relu_device) {
    testRelu(true);
}

TEST(autograd, test_sigmoid_device) {
    testSigmoid(true);
}

TEST(autograd, test_combine_device) {
    testCombine(true);
}

TEST(autograd, test_fork_device) {
    testFork(true);
}

#endif