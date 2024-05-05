/**
 * @file function.h
 * @brief
 *
 * @author Jan Warchocki
 * @date 03 May 2024
 *
 */

#ifndef NNLIB_FUNCTIONS_H
#define NNLIB_FUNCTIONS_H

#include <utility>

#include "tensor.h"

class Function {
public:
    std::vector<sTensor> parents;

    Function() = default;

    sTensor forward(const std::vector<sTensor>& args) {
        parents = args;
        sTensor result = forwardFn(args);
        for (auto& parent : parents) {
            if (parent->requiresGrad) {
                result->requiresGrad = true;
                break;
            }
        }
        return result;
    }

    std::vector<sTensor> backward(sTensor grad) {
        return backwardFn(std::move(grad));
    }

    virtual sTensor forwardFn(const std::vector<sTensor>& args) = 0;

    virtual std::vector<sTensor> backwardFn(sTensor grad) = 0;

    virtual ~Function() = default;
};

class SumReduce : public Function {
    std::vector<size_t> shapeCache;
public:
    sTensor forwardFn(const std::vector<sTensor>& args) override;

    std::vector<sTensor> backwardFn(sTensor grad) override;
};

class Add : public Function {
public:
    sTensor forwardFn(const std::vector<sTensor>& args) override;

    std::vector<sTensor> backwardFn(sTensor grad) override;
};

class Subtract : public Function {
public:
    sTensor forwardFn(const std::vector<sTensor>& args) override;

    std::vector<sTensor> backwardFn(sTensor grad) override;
};

class Hadamard : public Function {
    sTensor cacheA;
    sTensor cacheB;
public:
    sTensor forwardFn(const std::vector<sTensor>& args) override;

    std::vector<sTensor> backwardFn(sTensor grad) override;
};

class Divide : public Function {
    sTensor cacheA;
    sTensor cacheB;
public:
    sTensor forwardFn(const std::vector<sTensor>& args) override;

    std::vector<sTensor> backwardFn(sTensor grad) override;
};

class Log : public Function {
    sTensor cacheA;
public:
    sTensor forwardFn(const std::vector<sTensor>& args) override;

    std::vector<sTensor> backwardFn(sTensor grad) override;
};

class MatVecMul : public Function {
    sTensor cacheA;
    sTensor cacheB;
public:
    sTensor forwardFn(const std::vector<sTensor>& args) override;

    std::vector<sTensor> backwardFn(sTensor grad) override;
};

class Matmul : public Function {
    sTensor cacheA;
    sTensor cacheB;
public:
    sTensor forwardFn(const std::vector<sTensor>& args) override;

    std::vector<sTensor> backwardFn(sTensor grad) override;

    ~Matmul() override = default;
};

class Transpose : public Function {
    sTensor cacheA;
public:
    sTensor forwardFn(const std::vector<sTensor>& args) override;

    std::vector<sTensor> backwardFn(sTensor grad) override;
};


#endif //NNLIB_FUNCTIONS_H
