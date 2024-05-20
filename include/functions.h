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
#include <runtime.h>
#include "tensor.h"
#include "tuple_utils.h"
#include <exceptions.h>
#include "location_verifiers.h"
#include "tensor_operations_on_device.cuh"
#include "tensor_operations_on_host.h"

template<typename T>
void fill(float value, std::shared_ptr<Tensor<T>>& tensor) {
    if (tensor->location == HOST) {
        fillTensorOnHost(*tensor, value);
    } else {
        fillTensorOnDevice(*tensor, value);
    }
}

template<typename T>
void fill(const std::shared_ptr<Tensor<T>>& value, std::shared_ptr<Tensor<T>>& tensor) {
    if (tensor->location == HOST) {
        fillTensorOnHost(*tensor, *value);
    } else {
        fillTensorOnDevice(*tensor, *value);
    }
}

namespace no_grad {
    template<typename T>
    std::shared_ptr<Tensor<T>> sum(const std::shared_ptr<Tensor<T>>& a) {
        std::vector<size_t> shape = {1};
        auto result = std::make_shared<Tensor<T>>(shape, a->location);
        ::fill(0.0f, result);
        if (a->location == HOST) {
            sumTensorOnHost(*a, *result);
        } else {
            sumTensorOnDevice(*a, *result);
        }
        return result;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> addTensors(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
        if (a->shape != b->shape) {
            throw SizeMismatchException();
        }

        auto result = std::make_shared<Tensor<T>>(a->shape, a->location);

        std::initializer_list<DataLocation> locations = {a->location, b->location};
        if (allLocationsAreHost(locations)) {
            addTensorsOnHost(*a, *b, *result);
        } else if (allLocationsAreDevice(locations)) {
            addTensorsOnDevice(*a, *b, *result);
        } else {
            throw DifferentDataLocationException();
        }

        return result;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> addBroadcast(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
        if (a->shape[1] != b->shape[0]) {
            throw SizeMismatchException();
        }

        auto result = std::make_shared<Tensor<T>>(a->shape, a->location);

        std::initializer_list<DataLocation> locations = {a->location, b->location};
        if (allLocationsAreHost(locations)) {
            addBroadcastOnHost(*a, *b, *result);
        } else if (allLocationsAreDevice(locations)) {
            addBroadcastOnDevice(*a, *b, *result);
        } else {
            throw DifferentDataLocationException();
        }

        return result;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> add(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
        if (a->shape.size() == 2 && b->shape.size() == 1) {
            return no_grad::addBroadcast(a, b);
        } else {
            return no_grad::addTensors(a, b);
        }
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> subtract(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
        if (a->shape != b->shape) {
            throw SizeMismatchException();
        }

        std::shared_ptr<Tensor<T>> result = std::make_shared<Tensor<T>>(a->shape, a->location);

        std::initializer_list<DataLocation> locations = {a->location, b->location};
        if (allLocationsAreHost(locations)) {
            subtractTensorsOnHost(*a, *b, *result);
        } else if (allLocationsAreDevice(locations)) {
            subtractTensorsOnDevice(*a, *b, *result);
        } else {
            throw DifferentDataLocationException();
        }

        return result;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> hadamard(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
        if (a->shape != b->shape) {
            throw SizeMismatchException();
        }

        auto result = std::make_shared<Tensor<T>>(a->shape, a->location);

        std::initializer_list<DataLocation> locations = {a->location, b->location};
        if (allLocationsAreHost(locations)) {
            hadamardTensorsOnHost(*a, *b, *result);
        } else if (allLocationsAreDevice(locations)) {
            hadamardTensorsOnDevice(*a, *b, *result);
        } else {
            throw DifferentDataLocationException();
        }

        return result;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> divide(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
        if (a->shape != b->shape) {
            throw SizeMismatchException();
        }

        auto result = std::make_shared<Tensor<T>>(a->shape, a->location);

        std::initializer_list<DataLocation> locations = {a->location, b->location};
        if (allLocationsAreHost(locations)) {
            divideTensorsOnHost(*a, *b, *result);
        } else if (allLocationsAreDevice(locations)) {
            divideTensorsOnDevice(*a, *b, *result);
        } else {
            throw DifferentDataLocationException();
        }

        return result;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> log(const std::shared_ptr<Tensor<T>>& a) {
        auto result = std::make_shared<Tensor<T>>(a->shape, a->location);

        std::initializer_list<DataLocation> locations = {a->location};
        if (allLocationsAreHost(locations)) {
            logTensorOnHost(*a, *result);
        } else if (allLocationsAreDevice(locations)) {
            logTensorOnDevice(*a, *result);
        } else {
            throw DifferentDataLocationException();
        }

        return result;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> multiply(const std::shared_ptr<Tensor<T>>& a, float constant) {
        auto result = std::make_shared<Tensor<T>>(a->shape, a->location);

        std::initializer_list<DataLocation> locations = {a->location};
        if (allLocationsAreHost(locations)) {
            multiplyTensorOnHost(*a, constant, *result);
        } else if (allLocationsAreDevice(locations)) {
            multiplyTensorOnDevice(*a, constant, *result);
        } else {
            throw DifferentDataLocationException();
        }

        return result;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> matvecmul(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
        if (a->shape[1] != b->shape[0]) {
            throw SizeMismatchException();
        }

        std::vector<size_t> shape = {a->shape[0]};
        auto result = std::make_shared<Tensor<T>>(shape, a->location);

        std::initializer_list<DataLocation> locations = {a->location, b->location};
        if (allLocationsAreHost(locations)) {
            multiplyMatrixVectorOnHost(*a, *b, *result);
        } else if (allLocationsAreDevice(locations)) {
            multiplyMatrixVectorOnDevice(*a, *b, *result);
        } else {
            throw DifferentDataLocationException();
        }

        return result;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> matmul(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
        if (a->shape[1] != b->shape[0]) {
            throw SizeMismatchException();
        }

        std::vector<size_t> shape = {a->shape[0], b->shape[1]};
        auto result = std::make_shared<Tensor<T>>(shape, a->location);

        std::initializer_list<DataLocation> locations = {a->location, b->location};
        if (allLocationsAreHost(locations)) {
            multiplyMatrixMatrixOnHost(*a, *b, *result);
        } else if (allLocationsAreDevice(locations)) {
            multiplyMatrixMatrixOnDevice(*a, *b, *result);
        } else {
            throw DifferentDataLocationException();
        }

        return result;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> multiply(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
        if (a->shape.size() == 2 && b->shape.size() == 2) {
            return no_grad::matmul(a, b);
        } else if (a->shape.size() == 2 && b->shape.size() == 1) {
            return no_grad::matvecmul(a, b);
        } else {
            throw UnsupportedOperationException();
        }
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> transpose(const std::shared_ptr<Tensor<T>>& a) {
        std::vector<size_t> shape = {a->shape[1], a->shape[0]};
        auto result = std::make_shared<Tensor<T>>(shape, a->location);

        std::initializer_list<DataLocation> locations = {a->location};
        if (allLocationsAreHost(locations)) {
            transposeMatrixOnHost(*a, *result);
        } else if (allLocationsAreDevice(locations)) {
            transposeMatrixOnDevice(*a, *result);
        } else {
            throw DifferentDataLocationException();
        }

        return result;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> relu(const std::shared_ptr<Tensor<T>>& a) {
        auto result = std::make_shared<Tensor<T>>(a->shape, a->location);

        std::initializer_list<DataLocation> locations = {a->location};
        if (allLocationsAreHost(locations)) {
            reluTensorOnHost(*a, *result);
        } else if (allLocationsAreDevice(locations)) {
            reluTensorOnDevice(*a, *result);
        } else {
            throw DifferentDataLocationException();
        }

        return result;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> sigmoid(const std::shared_ptr<Tensor<T>>& a) {
        auto result = std::make_shared<Tensor<T>>(a->shape, a->location);

        std::initializer_list<DataLocation> locations = {a->location};
        if (allLocationsAreHost(locations)) {
            sigmoidTensorOnHost(*a, *result);
        } else if (allLocationsAreDevice(locations)) {
            sigmoidTensorOnDevice(*a, *result);
        } else {
            throw DifferentDataLocationException();
        }

        return result;
    }
} // namespace no_grad





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
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a) override {
        shapeCache = a->shape;
        auto result = no_grad::sum(a);
        return result;
    }

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override {
        if (!this->parents[0]->requiresGrad) {
            return {nullptr};
        }

        auto gradA = std::make_shared<Tensor<T>>(shapeCache, grad->location);
        fill(grad, gradA);
        return {gradA};
    }
};

template<typename T>
class Add : public Function<T, std::shared_ptr<Tensor<T>>, std::shared_ptr<Tensor<T>>> {
public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) override {
        return no_grad::addTensors(a, b);
    }

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override {
        auto gradA = this->parents[0]->requiresGrad ? grad->copy() : nullptr;
        auto gradB = this->parents[1]->requiresGrad ? grad->copy() : nullptr;
        return {gradA, gradB};
    }
};

template<typename T>
class AddBroadcast : public Function<T, std::shared_ptr<Tensor<T>>, std::shared_ptr<Tensor<T>>> {
public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) override {
        return no_grad::addBroadcast(a, b);
    }

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override {
        auto gradA = this->parents[0]->requiresGrad ? grad->copy() : nullptr;

        std::shared_ptr<Tensor<T>> gradB = nullptr;
        if (this->parents[1]->requiresGrad) {
            std::vector<size_t> shape = {grad->shape[0]};
            auto ones = std::make_shared<Tensor<T>>(shape, grad->location);
            fill(1.0f, ones);
            gradB = no_grad::multiply(no_grad::transpose(grad), ones); // TODO: replace later with sum reduction
        }
        return {gradA, gradB};
    }
};

template<typename T>
class Subtract : public Function<T, std::shared_ptr<Tensor<T>>, std::shared_ptr<Tensor<T>>> {
public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) override {
        return no_grad::subtract(a, b);
    }

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override {
        auto gradA = this->parents[0]->requiresGrad ? grad->copy() : nullptr;
        auto gradB = this->parents[1]->requiresGrad ? no_grad::multiply(grad, -1.0f) : nullptr;
        return {gradA, gradB};
    }
};

template<typename T>
class Hadamard : public Function<T, std::shared_ptr<Tensor<T>>, std::shared_ptr<Tensor<T>>> {
    std::shared_ptr<Tensor<T>> cacheA;
    std::shared_ptr<Tensor<T>> cacheB;

public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) override {
        // a and b flipped because gradient of a uses b and vice-versa
        if (b->requiresGrad) {
            cacheA = a->copy();
        }
        if (a->requiresGrad) {
            cacheB = b->copy();
        }
        return no_grad::hadamard(a, b);
    }

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override {
        auto gradA = this->parents[0]->requiresGrad ? no_grad::hadamard(grad, cacheB) : nullptr;
        auto gradB = this->parents[1]->requiresGrad ? no_grad::hadamard(grad, cacheA) : nullptr;
        return {gradA, gradB};
    }
};

template<typename T>
class Divide : public Function<T, std::shared_ptr<Tensor<T>>, std::shared_ptr<Tensor<T>>> {
    std::shared_ptr<Tensor<T>> cacheA;
    std::shared_ptr<Tensor<T>> cacheB;

public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) override {
        if (b->requiresGrad) {
            cacheA = a->copy();
        }
        if (a->requiresGrad || b->requiresGrad) {
            cacheB = b->copy();
        }
        return no_grad::divide(a, b);
    }

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override {
        auto gradA = this->parents[0]->requiresGrad ? no_grad::divide(grad, cacheB) : nullptr;

        std::shared_ptr<Tensor<T>> gradB = nullptr;
        if (this->parents[1]->requiresGrad) {
            gradB = no_grad::divide(no_grad::hadamard(grad, cacheA), no_grad::hadamard(cacheB, cacheB));
            gradB = no_grad::multiply(gradB, -1.0f);
        }
        return {gradA, gradB};
    }
};

template<typename T>
class Log : public Function<T, std::shared_ptr<Tensor<T>>> {
    std::shared_ptr<Tensor<T>> cacheA;

public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a) override {
        if (a->requiresGrad) {
            cacheA = a->copy();
        }
        return no_grad::log(a);
    }

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override {
        auto gradA = this->parents[0]->requiresGrad ? no_grad::divide(grad, cacheA) : nullptr;
        return {gradA};
    }
};

template<typename T>
class MulConstant : public Function<T, std::shared_ptr<Tensor<T>>, float> {
    float constantCache;

public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a, const float& b) override {
        constantCache = b;
        return no_grad::multiply(a, b);
    }

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override {
        auto gradA = this->parents[0]->requiresGrad ? no_grad::multiply(grad, constantCache) : nullptr;
        return {gradA};
    }
};

template<typename T>
class MatVecMul : public Function<T, std::shared_ptr<Tensor<T>>, std::shared_ptr<Tensor<T>>> {
    std::shared_ptr<Tensor<T>> cacheA;
    std::shared_ptr<Tensor<T>> cacheB;

public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) override {
        // a and b flipped because gradient of a uses b and vice-versa
        if (b->requiresGrad) {
            cacheA = a->copy();
        }
        if (a->requiresGrad) {
            cacheB = b->copy();
        }
        return no_grad::matvecmul(a, b);
    }

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override {
        cacheB->shape = {cacheB->shape[0], 1};
        grad->shape = {grad->shape[0], 1};

        auto gradA = this->parents[0]->requiresGrad ? no_grad::multiply(grad, no_grad::transpose(cacheB)) : nullptr;
        std::shared_ptr<Tensor<T>> gradB = nullptr;
        if (this->parents[1]->requiresGrad) {
            gradB = no_grad::multiply(no_grad::transpose(cacheA), grad);
            gradB->shape = {gradB->shape[0]};
        }
        return {gradA, gradB};
    }
};

template<typename T>
class Matmul : public Function<T, std::shared_ptr<Tensor<T>>, std::shared_ptr<Tensor<T>>> {
    std::shared_ptr<Tensor<T>> cacheA;
    std::shared_ptr<Tensor<T>> cacheB;

public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) override {
        // a and b flipped because gradient of a uses b and vice-versa
        if (b->requiresGrad) {
            cacheA = a->copy();
        }
        if (a->requiresGrad) {
            cacheB = b->copy();
        }
        return no_grad::matmul(a, b);
    }

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override {
        auto gradA = this->parents[0]->requiresGrad ? no_grad::multiply(grad, no_grad::transpose(cacheB)) : nullptr;
        auto gradB = this->parents[1]->requiresGrad ? no_grad::multiply(no_grad::transpose(cacheA), grad) : nullptr;
        return {gradA, gradB};
    }

    ~Matmul() override = default;
};

template<typename T>
class Transpose : public Function<T, std::shared_ptr<Tensor<T>>> {
    std::shared_ptr<Tensor<T>> cacheA;

public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a) override {
        if (a->requiresGrad) {
            cacheA = a->copy();
        }
        return no_grad::transpose(a);
    }

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override {
        auto gradA = this->parents[0]->requiresGrad ? no_grad::transpose(grad) : nullptr;
        return {gradA};
    }
};

template<typename T>
class ReLU : public Function<T, std::shared_ptr<Tensor<T>>> {
    std::shared_ptr<Tensor<T>> cacheA;

public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a) override {
        if (a->requiresGrad) {
            cacheA = a->copy();
        }
        return no_grad::relu(a);
    }

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override {
        if (!this->parents[0]->requiresGrad) {
            return {nullptr};
        }

        auto gradA = std::make_shared<Tensor<T>>(cacheA->shape, cacheA->location);

        std::initializer_list<DataLocation> locations = {cacheA->location};
        if (allLocationsAreHost(locations)) {
            reluDerivativeTensorOnHost(*cacheA, *gradA);
        } else if (allLocationsAreDevice(locations)) {
            reluDerivativeTensorOnDevice(*cacheA, *gradA);
        } else {
            throw DifferentDataLocationException();
        }
        gradA = hadamard(grad, gradA);
        return {gradA};
    }
};

template<typename T>
class Sigmoid : public Function<T, std::shared_ptr<Tensor<T>>> {
    std::shared_ptr<Tensor<T>> cacheA;

public:
    std::shared_ptr<Tensor<T>> forwardFn(const std::shared_ptr<Tensor<T>>& a) override {
        auto result = no_grad::sigmoid(a);
        if (a->requiresGrad) {
            cacheA = result->copy();
        }
        return result;
    }

    std::vector<std::shared_ptr<Tensor<T>>> backwardFn(std::shared_ptr<Tensor<T>> grad) override {
        if (!this->parents[0]->requiresGrad) {
            return {nullptr};
        }

        auto ones = std::make_shared<Tensor<T>>(cacheA->shape, cacheA->location);
        fill(1.0f, ones);

        auto gradA = no_grad::hadamard(grad, no_grad::hadamard(cacheA, no_grad::subtract(ones, cacheA)));
        return {gradA};
    }
};

template<typename T>
std::shared_ptr<Tensor<T>> sum(const std::shared_ptr<Tensor<T>>& a) {
    if (Runtime::getInstance().useGradient) {
        auto sum = std::make_shared<SumReduce<T>>();
        auto result = sum->forward(a);
        result->gradFunction = sum;
        return result;
    } else {
        return no_grad::sum(a);
    }
}

template<typename T>
std::shared_ptr<Tensor<T>> addTensors(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
    if (Runtime::getInstance().useGradient) {
        auto add = std::make_shared<Add<T>>();
        auto result = add->forward(a, b);
        result->gradFunction = add;
        return result;
    } else {
        return no_grad::addTensors(a, b);
    }
}

template<typename T>
std::shared_ptr<Tensor<T>> addBroadcast(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
    if (Runtime::getInstance().useGradient) {
        auto add = std::make_shared<AddBroadcast<T>>();
        auto result = add->forward(a, b);
        result->gradFunction = add;
        return result;
    } else {
        return no_grad::addBroadcast(a, b);
    }
}

template<typename T>
std::shared_ptr<Tensor<T>> add(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
    if (a->shape.size() == 2 && b->shape.size() == 1) {
        return addBroadcast(a, b);
    } else {
        return addTensors(a, b);
    }
}

template<typename T>
std::shared_ptr<Tensor<T>> subtract(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
    if (Runtime::getInstance().useGradient) {
        auto subtract = std::make_shared<Subtract<T>>();
        auto result = subtract->forward(a, b);
        result->gradFunction = subtract;
        return result;
    } else {
        return no_grad::subtract(a, b);
    }
}

template<typename T>
std::shared_ptr<Tensor<T>> hadamard(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
    if (Runtime::getInstance().useGradient) {
        auto hadamard = std::make_shared<Hadamard<T>>();
        auto result = hadamard->forward(a, b);
        result->gradFunction = hadamard;
        return result;
    } else {
        return no_grad::hadamard(a, b);
    }
}

template<typename T>
std::shared_ptr<Tensor<T>> divide(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
    if (Runtime::getInstance().useGradient) {
        auto divide = std::make_shared<Divide<T>>();
        auto result = divide->forward(a, b);
        result->gradFunction = divide;
        return result;
    } else {
        return no_grad::divide(a, b);
    }
}

template<typename T>
std::shared_ptr<Tensor<T>> log(const std::shared_ptr<Tensor<T>>& a) {
    if (Runtime::getInstance().useGradient) {
        auto log = std::make_shared<Log<T>>();
        auto result = log->forward(a);
        result->gradFunction = log;
        return result;
    } else {
        return no_grad::log(a);
    }
}

template<typename T>
std::shared_ptr<Tensor<T>> multiply(const std::shared_ptr<Tensor<T>>& a, float constant) {
    if (Runtime::getInstance().useGradient) {
        auto multiply = std::make_shared<MulConstant<T>>();
        auto result = multiply->forward(a, constant);
        result->gradFunction = multiply;
        return result;
    } else {
        return no_grad::multiply(a, constant);
    }
}

template<typename T>
std::shared_ptr<Tensor<T>> matvecmul(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
    if (Runtime::getInstance().useGradient) {
        auto matvecmul = std::make_shared<MatVecMul<T>>();
        auto result = matvecmul->forward(a, b);
        result->gradFunction = matvecmul;
        return result;
    } else {
        return no_grad::matvecmul(a, b);
    }
}

template<typename T>
std::shared_ptr<Tensor<T>> matmul(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
    if (Runtime::getInstance().useGradient) {
        auto matmul = std::make_shared<Matmul<T>>();
        auto result = matmul->forward(a, b);
        result->gradFunction = matmul;
        return result;
    } else {
        return no_grad::matmul(a, b);
    }
}

template<typename T>
std::shared_ptr<Tensor<T>> multiply(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
    if (a->shape.size() == 2 && b->shape.size() == 2) {
        return matmul(a, b);
    } else if (a->shape.size() == 2 && b->shape.size() == 1) {
        return matvecmul(a, b);
    } else {
        throw UnsupportedOperationException();
    }
}

template<typename T>
std::shared_ptr<Tensor<T>> transpose(const std::shared_ptr<Tensor<T>>& a) {
    if (Runtime::getInstance().useGradient) {
        auto transpose = std::make_shared<Transpose<T>>();
        auto result = transpose->forward(a);
        result->gradFunction = transpose;
        return result;
    } else {
        return no_grad::transpose(a);
    }
}

template<typename T>
std::shared_ptr<Tensor<T>> relu(const std::shared_ptr<Tensor<T>>& a) {
    if (Runtime::getInstance().useGradient) {
        auto relu = std::make_shared<ReLU<T>>();
        auto result = relu->forward(a);
        result->gradFunction = relu;
        return result;
    } else {
        return no_grad::relu(a);
    }
}

template<typename T>
std::shared_ptr<Tensor<T>> sigmoid(const std::shared_ptr<Tensor<T>>& a) {
    if (Runtime::getInstance().useGradient) {
        auto sigmoid = std::make_shared<Sigmoid<T>>();
        auto result = sigmoid->forward(a);
        result->gradFunction = sigmoid;
        return result;
    } else {
        return no_grad::sigmoid(a);
    }
}

#endif //NNLIB_FUNCTIONS_H
