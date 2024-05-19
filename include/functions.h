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

#include <typeinfo>
#include <utility>

#include "tensor.h"
#include "tuple_utils.h"

template<typename T>
class BackwardFunction {
public:
    std::vector<std::shared_ptr<Tensor<T>>> parents;

    virtual std::vector<std::shared_ptr<Tensor<T>>> backward(std::shared_ptr<Tensor<T>> grad) = 0;

    virtual ~BackwardFunction() = default;
};

template<typename T, typename... Types>
class Function : public BackwardFunction<T> {
public:
    Function() = default;

    std::shared_ptr<Tensor<T>> forward(const Types&... args) {
        auto tup = getType<std::shared_ptr<Tensor<T>>>(std::make_tuple(args...));
        this->parents = toVector(tup);

        auto result = forwardFn(args...);
        for (auto& parent : this->parents) {
            if (parent->requiresGrad) {
                result->requiresGrad = true;
                break;
            }
        }
        return result;
    }

    std::vector<std::shared_ptr<Tensor<T>>> backward(std::shared_ptr<Tensor<T>> grad) override {
        return backwardFn(std::move(grad));
    }

    virtual std::shared_ptr<Tensor<T>> forwardFn(const Types&... args) = 0;

    virtual std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) = 0;

    ~Function() override = default;
};

template<typename T>
class SumReduce : public Function<T, std::shared_ptr<Tensor<T>>> {
    std::vector<size_t> shapeCache;

public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a) override;

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override;
};

template<typename T>
class Add : public Function<T, std::shared_ptr<Tensor<T>>, std::shared_ptr<Tensor<T>>> {
public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) override;

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override;
};

template<typename T>
class AddBroadcast : public Function<T, std::shared_ptr<Tensor<T>>, std::shared_ptr<Tensor<T>>> {
public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) override;

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override;
};

template<typename T>
class Subtract : public Function<T, std::shared_ptr<Tensor<T>>, std::shared_ptr<Tensor<T>>> {
public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) override;

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override;
};

template<typename T>
class Hadamard : public Function<T, std::shared_ptr<Tensor<T>>, std::shared_ptr<Tensor<T>>> {
    std::shared_ptr<Tensor<T>> cacheA;
    std::shared_ptr<Tensor<T>> cacheB;

public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) override;

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override;
};

template<typename T>
class Divide : public Function<T, std::shared_ptr<Tensor<T>>, std::shared_ptr<Tensor<T>>> {
    std::shared_ptr<Tensor<T>> cacheA;
    std::shared_ptr<Tensor<T>> cacheB;

public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) override;

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override;
};

template<typename T>
class Log : public Function<T, std::shared_ptr<Tensor<T>>> {
    std::shared_ptr<Tensor<T>> cacheA;

public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& args) override;

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override;
};

template<typename T>
class MulConstant : public Function<T, std::shared_ptr<Tensor<T>>, float> {
    float constantCache;

public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a, const float& b) override;

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override;
};

template<typename T>
class MatVecMul : public Function<T, std::shared_ptr<Tensor<T>>, std::shared_ptr<Tensor<T>>> {
    std::shared_ptr<Tensor<T>> cacheA;
    std::shared_ptr<Tensor<T>> cacheB;

public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) override;

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override;
};

template<typename T>
class Matmul : public Function<T, std::shared_ptr<Tensor<T>>, std::shared_ptr<Tensor<T>>> {
    std::shared_ptr<Tensor<T>> cacheA;
    std::shared_ptr<Tensor<T>> cacheB;

public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) override;

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override;

    ~Matmul() override = default;
};

template<typename T>
class Transpose : public Function<T, std::shared_ptr<Tensor<T>>> {
    std::shared_ptr<Tensor<T>> cacheA;

public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a) override;

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override;
};

template<typename T>
class ReLU : public Function<T, std::shared_ptr<Tensor<T>>> {
    std::shared_ptr<Tensor<T>> cacheA;

public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a) override;

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override;
};

template<typename T>
class Sigmoid : public Function<T, std::shared_ptr<Tensor<T>>> {
    std::shared_ptr<Tensor<T>> cacheA;

public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a) override;

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override;
};

#endif //NNLIB_FUNCTIONS_H
