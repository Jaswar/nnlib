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
#include <typeinfo>

#include "tensor.h"
#include "tuple_utils.h"


class BackwardFunction {
public:
    std::vector<sTensor> parents;

    virtual std::vector<sTensor> backward(sTensor grad) = 0;

    virtual ~BackwardFunction() = default;
};

template<typename... Types>
class Function : public BackwardFunction {
public:
    Function() = default;

    sTensor forward(const Types&... args) {
        auto tup = get_type<sTensor>(std::make_tuple(args...));
        parents = to_vector(tup);

        sTensor result = forwardFn(args...);
        for (auto& parent : parents) {
            if (parent->requiresGrad) {
                result->requiresGrad = true;
                break;
            }
        }
        return result;
    }

    std::vector<sTensor> backward(sTensor grad) override {
        return backwardFn(std::move(grad));
    }

    virtual sTensor forwardFn(const Types&... args) = 0;

    virtual std::vector<sTensor> backwardFn(sTensor grad) = 0;

    ~Function() override = default;
};

class SumReduce : public Function<sTensor> {
    std::vector<size_t> shapeCache;
public:
    sTensor forwardFn(const sTensor& a) override;

    std::vector<sTensor> backwardFn(sTensor grad) override;
};

class Add : public Function<sTensor, sTensor> {
public:
    sTensor forwardFn(const sTensor& a, const sTensor& b) override;

    std::vector<sTensor> backwardFn(sTensor grad) override;
};

class AddBroadcast : public Function<sTensor, sTensor> {
public:
    sTensor forwardFn(const sTensor& a, const sTensor& b) override;

    std::vector<sTensor> backwardFn(sTensor grad) override;
};

class Subtract : public Function<sTensor, sTensor> {
public:
    sTensor forwardFn(const sTensor& a, const sTensor& b) override;

    std::vector<sTensor> backwardFn(sTensor grad) override;
};

class Hadamard : public Function<sTensor, sTensor> {
    sTensor cacheA;
    sTensor cacheB;
public:
    sTensor forwardFn(const sTensor& a, const sTensor& b) override;

    std::vector<sTensor> backwardFn(sTensor grad) override;
};

class Divide : public Function<sTensor, sTensor> {
    sTensor cacheA;
    sTensor cacheB;
public:
    sTensor forwardFn(const sTensor& a, const sTensor& b) override;

    std::vector<sTensor> backwardFn(sTensor grad) override;
};

class Log : public Function<sTensor> {
    sTensor cacheA;
public:
    sTensor forwardFn(const sTensor& args) override;

    std::vector<sTensor> backwardFn(sTensor grad) override;
};

class MulConstant : public Function<sTensor, float> {
    float constantCache;
public:
    sTensor forwardFn(const sTensor& a, const float& b) override;

    std::vector<sTensor> backwardFn(sTensor grad) override;
};

class MatVecMul : public Function<sTensor, sTensor> {
    sTensor cacheA;
    sTensor cacheB;
public:
    sTensor forwardFn(const sTensor& a, const sTensor& b) override;

    std::vector<sTensor> backwardFn(sTensor grad) override;
};

class Matmul : public Function<sTensor, sTensor> {
    sTensor cacheA;
    sTensor cacheB;
public:
    sTensor forwardFn(const sTensor& a, const sTensor& b) override;

    std::vector<sTensor> backwardFn(sTensor grad) override;

    ~Matmul() override = default;
};

class Transpose : public Function<sTensor> {
    sTensor cacheA;
public:
    sTensor forwardFn(const sTensor& a) override;

    std::vector<sTensor> backwardFn(sTensor grad) override;
};

class ReLU : public Function<sTensor> {
    sTensor cacheA;
public:
    sTensor forwardFn(const sTensor& a) override;

    std::vector<sTensor> backwardFn(sTensor grad) override;
};

class Sigmoid : public Function<sTensor> {
    sTensor cacheA;
public:
    sTensor forwardFn(const sTensor& a) override;

    std::vector<sTensor> backwardFn(sTensor grad) override;
};

#endif //NNLIB_FUNCTIONS_H
