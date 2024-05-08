/**
 * @file function.cpp
 * @brief
 *
 * @author Jan Warchocki
 * @date 03 May 2024
 *
 */

#include "functions.h"
#include "exceptions/different_data_location_exception.h"
#include "exceptions/size_mismatch_exception.h"
#include "exceptions/unsupported_operation_exception.h"
#include "runtime.h"
#include "tensor.h"
#include "tensor_operations_on_device.cuh"
#include "tensor_operations_on_host.h"
#include "utils/location_verifiers.h"


void fill(float value, sTensor& tensor) {
    if (tensor->location == HOST) {
        fillTensorOnHost(*tensor, value);
    } else {
        fillTensorOnDevice(*tensor, value);
    }
}

void fill(const sTensor& value, sTensor& destination) {
    if (destination->location == HOST) {
        fillTensorOnHost(*destination, *value);
    } else {
        fillTensorOnDevice(*destination, *value);
    }
}

sTensor no_grad::sum(const sTensor& a) {
    std::vector<size_t> shape = {1};
    sTensor result = std::make_shared<Tensor>(shape, a->location);
    ::fill(0.0f, result);
    if (a->location == HOST) {
        sumTensorOnHost(*a, *result);
    } else {
        sumTensorOnDevice(*a, *result);
    }
    return result;
}

sTensor sum(const sTensor& a) {
    if (Runtime::getInstance().useGradient) {
        auto sum = std::make_shared<SumReduce>();
        sTensor result = sum->forward(a);
        result->gradFunction = sum;
        return result;
    } else {
        return no_grad::sum(a);
    }
}

sTensor SumReduce::forwardFn(const sTensor& a) {
    shapeCache = a->shape;
    sTensor result = no_grad::sum(a);
    return result;
}

std::vector<sTensor> SumReduce::backwardFn(sTensor grad) {
    if (!parents[0]->requiresGrad) {
        return {nullptr};
    }

    sTensor gradA = std::make_shared<Tensor>(shapeCache, grad->location);
    fill(grad, gradA);
    return {gradA};
}

namespace no_grad {
    sTensor addTensors(const sTensor& a, const sTensor& b) {
        if (a->shape != b->shape) {
            throw SizeMismatchException();
        }

        sTensor result = std::make_shared<Tensor>(a->shape, a->location);

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

    sTensor addBroadcast(const sTensor& a, const sTensor& b) {
        if (a->shape[1] != b->shape[0]) {
            throw SizeMismatchException();
        }

        sTensor result = std::make_shared<Tensor>(a->shape, a->location);

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
}

sTensor no_grad::add(const sTensor& a, const sTensor& b) {
    if (a->shape.size() == 2 && b->shape.size() == 1) {
        return no_grad::addBroadcast(a, b);
    } else {
        return no_grad::addTensors(a, b);
    }
}

sTensor addTensors(const sTensor& a, const sTensor& b) {
    if (Runtime::getInstance().useGradient) {
        auto add = std::make_shared<Add>();
        sTensor result = add->forward(a, b);
        result->gradFunction = add;
        return result;
    } else {
        return no_grad::addTensors(a, b);
    }
}

sTensor addBroadcast(const sTensor& a, const sTensor& b) {
    if (Runtime::getInstance().useGradient) {
        auto add = std::make_shared<AddBroadcast>();
        sTensor result = add->forward(a, b);
        result->gradFunction = add;
        return result;
    } else {
        return no_grad::addBroadcast(a, b);
    }
}

sTensor add(const sTensor& a, const sTensor& b) {
    if (a->shape.size() == 2 && b->shape.size() == 1) {
        return addBroadcast(a, b);
    } else {
        return addTensors(a, b);
    }
}

sTensor Add::forwardFn(const sTensor& a, const sTensor& b) {
    return no_grad::addTensors(a, b);
}

std::vector<sTensor> Add::backwardFn(sTensor grad) {
    sTensor gradA = parents[0]->requiresGrad ? grad->copy() : nullptr;
    sTensor gradB = parents[1]->requiresGrad ? grad->copy() : nullptr;
    return {gradA, gradB};
}

sTensor AddBroadcast::forwardFn(const sTensor& a, const sTensor& b) {
    return no_grad::addBroadcast(a, b);
}

std::vector<sTensor> AddBroadcast::backwardFn(sTensor grad) {
    sTensor gradA = parents[0]->requiresGrad ? grad->copy() : nullptr;

    sTensor gradB = nullptr;
    if (parents[1]->requiresGrad) {
        std::vector<size_t> shape = {grad->shape[0]};
        sTensor ones = std::make_shared<Tensor>(shape, grad->location);
        fill(1.0f, ones);
        gradB = no_grad::multiply(no_grad::transpose(grad), ones);  // TODO: replace later with sum reduction
    }
    return {gradA, gradB};
}

sTensor no_grad::subtract(const sTensor& a, const sTensor& b) {
    if (a->shape != b->shape) {
        throw SizeMismatchException();
    }

    sTensor result = std::make_shared<Tensor>(a->shape, a->location);

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

sTensor subtract(const sTensor& a, const sTensor& b) {
    if (Runtime::getInstance().useGradient) {
        auto subtract = std::make_shared<Subtract>();
        sTensor result = subtract->forward(a, b);
        result->gradFunction = subtract;
        return result;
    } else {
        return no_grad::subtract(a, b);
    }
}

sTensor Subtract::forwardFn(const sTensor& a, const sTensor& b) {
    return no_grad::subtract(a, b);
}

std::vector<sTensor> Subtract::backwardFn(sTensor grad) {
    sTensor gradA = parents[0]->requiresGrad ? grad->copy() : nullptr;
    sTensor gradB = parents[1]->requiresGrad ? no_grad::multiply(grad, -1.0f) : nullptr;
    return {gradA, gradB};
}

sTensor no_grad::hadamard(const sTensor& a, const sTensor& b) {
    if (a->shape != b->shape) {
        throw SizeMismatchException();
    }

    sTensor result = std::make_shared<Tensor>(a->shape, a->location);

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

sTensor hadamard(const sTensor& a, const sTensor& b) {
    if (Runtime::getInstance().useGradient) {
        auto hadamard = std::make_shared<Hadamard>();
        sTensor result = hadamard->forward(a, b);
        result->gradFunction = hadamard;
        return result;
    } else {
        return no_grad::hadamard(a, b);
    }
}

sTensor Hadamard::forwardFn(const sTensor& a, const sTensor& b) {
    // a and b flipped because gradient of a uses b and vice-versa
    if (b->requiresGrad) {
        cacheA = a->copy();
    }
    if (a->requiresGrad) {
        cacheB = b->copy();
    }
    return no_grad::hadamard(a, b);
}

std::vector<sTensor> Hadamard::backwardFn(sTensor grad) {
    sTensor gradA = parents[0]->requiresGrad ? no_grad::hadamard(grad, cacheB) : nullptr;
    sTensor gradB = parents[1]->requiresGrad ? no_grad::hadamard(grad, cacheA) : nullptr;
    return {gradA, gradB};
}


sTensor no_grad::divide(const sTensor& a, const sTensor& b) {
    if (a->shape != b->shape) {
        throw SizeMismatchException();
    }

    sTensor result = std::make_shared<Tensor>(a->shape, a->location);

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

sTensor divide(const sTensor& a, const sTensor& b) {
    if (Runtime::getInstance().useGradient) {
        auto divide = std::make_shared<Divide>();
        sTensor result = divide->forward(a, b);
        result->gradFunction = divide;
        return result;
    } else {
        return no_grad::divide(a, b);
    }
}

sTensor Divide::forwardFn(const sTensor& a, const sTensor& b) {
    if (b->requiresGrad) {
        cacheA = a->copy();
    }
    if (a->requiresGrad || b->requiresGrad) {
        cacheB = b->copy();
    }
    return no_grad::divide(a, b);
}

std::vector<sTensor> Divide::backwardFn(sTensor grad) {
    sTensor gradA = parents[0]->requiresGrad ? no_grad::divide(grad, cacheB) : nullptr;

    sTensor gradB = nullptr;
    if (parents[1]->requiresGrad) {
        gradB = no_grad::divide(no_grad::hadamard(grad, cacheA), no_grad::hadamard(cacheB, cacheB));
        gradB = no_grad::multiply(gradB, -1.0f);
    }
    return {gradA, gradB};
}

sTensor no_grad::log(const sTensor& a) {
    sTensor result = std::make_shared<Tensor>(a->shape, a->location);

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

sTensor log(const sTensor& a) {
    if (Runtime::getInstance().useGradient) {
        auto log = std::make_shared<Log>();
        sTensor result = log->forward(a);
        result->gradFunction = log;
        return result;
    } else {
        return no_grad::log(a);
    }
}

sTensor Log::forwardFn(const sTensor& a) {
    if (a->requiresGrad) {
        cacheA = a->copy();
    }
    return no_grad::log(a);
}

std::vector<sTensor> Log::backwardFn(sTensor grad) {
    sTensor gradA = parents[0]->requiresGrad ? no_grad::divide(grad, cacheA) : nullptr;
    return {gradA};
}

sTensor no_grad::multiply(const sTensor& a, float b) {
    sTensor result = std::make_shared<Tensor>(a->shape, a->location);

    std::initializer_list<DataLocation> locations = {a->location};
    if (allLocationsAreHost(locations)) {
        multiplyTensorOnHost(*a, b, *result);
    } else if (allLocationsAreDevice(locations)) {
        multiplyTensorOnDevice(*a, b, *result);
    } else {
        throw DifferentDataLocationException();
    }

    return result;
}

sTensor multiply(const sTensor& a, float constant) {
    if (Runtime::getInstance().useGradient) {
        auto multiply = std::make_shared<MulConstant>();
        sTensor result = multiply->forward(a, constant);
        result->gradFunction = multiply;
        return result;
    } else {
        return no_grad::multiply(a, constant);
    }
}

sTensor MulConstant::forwardFn(const sTensor& a, const float& b) {
    constantCache = b;
    return no_grad::multiply(a, b);
}

std::vector<sTensor> MulConstant::backwardFn(sTensor grad) {
    sTensor gradA = parents[0]->requiresGrad ? no_grad::multiply(grad, constantCache) : nullptr;
    return {gradA};
}

namespace no_grad {
    sTensor matvecmul(const sTensor& a, const sTensor& b) {
        if (a->shape[1] != b->shape[0]) {
            throw SizeMismatchException();
        }

        std::vector<size_t> shape = {a->shape[0]};
        sTensor result = std::make_shared<Tensor>(shape, a->location);

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

    sTensor matmul(const sTensor& a, const sTensor& b) {
        if (a->shape[1] != b->shape[0]) {
            throw SizeMismatchException();
        }

        std::vector<size_t> shape = {a->shape[0], b->shape[1]};
        sTensor result = std::make_shared<Tensor>(shape, a->location);

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
}

sTensor no_grad::multiply(const sTensor& a, const sTensor& b) {
    if (a->shape.size() == 2 && b->shape.size() == 2) {
        return no_grad::matmul(a, b);
    } else if (a->shape.size() == 2 && b->shape.size() == 1) {
        return no_grad::matvecmul(a, b);
    } else {
        throw UnsupportedOperationException();
    }
}

sTensor matvecmul(const sTensor& a, const sTensor& b) {
    if (Runtime::getInstance().useGradient) {
        auto matvecmul = std::make_shared<MatVecMul>();
        sTensor result = matvecmul->forward(a, b);
        result->gradFunction = matvecmul;
        return result;
    } else {
        return no_grad::matvecmul(a, b);
    }
}

sTensor matmul(const sTensor& a, const sTensor& b) {
    if (Runtime::getInstance().useGradient) {
        auto matmul = std::make_shared<Matmul>();
        sTensor result = matmul->forward(a, b);
        result->gradFunction = matmul;
        return result;
    } else {
        return no_grad::matmul(a, b);
    }
}

sTensor multiply(const sTensor& a, const sTensor& b) {
    if (a->shape.size() == 2 && b->shape.size() == 2) {
        return matmul(a, b);
    } else if (a->shape.size() == 2 && b->shape.size() == 1) {
        return matvecmul(a, b);
    } else {
        throw UnsupportedOperationException();
    }
}

sTensor MatVecMul::forwardFn(const sTensor& a, const sTensor& b) {
    // a and b flipped because gradient of a uses b and vice-versa
    if (b->requiresGrad) {
        cacheA = a->copy();
    }
    if (a->requiresGrad) {
        cacheB = b->copy();
    }
    return no_grad::matvecmul(a, b);
}

std::vector<sTensor> MatVecMul::backwardFn(sTensor grad) {
    cacheB->shape = {cacheB->shape[0], 1};
    grad->shape = {grad->shape[0], 1};

    sTensor gradA = parents[0]->requiresGrad ? no_grad::multiply(grad, no_grad::transpose(cacheB)) : nullptr;
    sTensor gradB = nullptr;
    if (parents[1]->requiresGrad) {
        gradB = no_grad::multiply(no_grad::transpose(cacheA), grad);
        gradB->shape = {gradB->shape[0]};
    }
    return {gradA, gradB};
}

sTensor Matmul::forwardFn(const sTensor& a, const sTensor& b) {
    // a and b flipped because gradient of a uses b and vice-versa
    if (b->requiresGrad) {
        cacheA = a->copy();
    }
    if (a->requiresGrad) {
        cacheB = b->copy();
    }
    return no_grad::matmul(a, b);
}

std::vector<sTensor> Matmul::backwardFn(sTensor grad) {
    sTensor gradA = parents[0]->requiresGrad ? no_grad::multiply(grad, no_grad::transpose(cacheB)) : nullptr;
    sTensor gradB = parents[1]->requiresGrad ? no_grad::multiply(no_grad::transpose(cacheA), grad) : nullptr;
    return {gradA, gradB};
}

sTensor no_grad::transpose(const sTensor& a) {
    std::vector<size_t> shape = {a->shape[1], a->shape[0]};
    sTensor result = std::make_shared<Tensor>(shape, a->location);

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

sTensor transpose(const sTensor& a) {
    if (Runtime::getInstance().useGradient) {
        auto transpose = std::make_shared<Transpose>();
        sTensor result = transpose->forward(a);
        result->gradFunction = transpose;
        return result;
    } else {
        return no_grad::transpose(a);
    }
}

sTensor Transpose::forwardFn(const sTensor& a) {
    if (a->requiresGrad) {
        cacheA = a->copy();
    }
    return no_grad::transpose(a);
}

std::vector<sTensor> Transpose::backwardFn(sTensor grad) {
    sTensor gradA = parents[0]->requiresGrad ? no_grad::transpose(grad) : nullptr;
    return {gradA};
}

sTensor no_grad::relu(const sTensor& a) {
    sTensor result = std::make_shared<Tensor>(a->shape, a->location);

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

sTensor relu(const sTensor& a) {
    if (Runtime::getInstance().useGradient) {
        auto relu = std::make_shared<ReLU>();
        sTensor result = relu->forward(a);
        result->gradFunction = relu;
        return result;
    } else {
        return no_grad::relu(a);
    }
}

sTensor ReLU::forwardFn(const sTensor& a) {
    if (a->requiresGrad) {
        cacheA = a->copy();
    }
    return no_grad::relu(a);
}

std::vector<sTensor> ReLU::backwardFn(sTensor grad) {
    if (!parents[0]->requiresGrad) {
        return {nullptr};
    }

    sTensor gradA = std::make_shared<Tensor>(cacheA->shape, cacheA->location);

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

sTensor no_grad::sigmoid(const sTensor& a) {
    sTensor result = std::make_shared<Tensor>(a->shape, a->location);

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

sTensor sigmoid(const sTensor& a) {
    if (Runtime::getInstance().useGradient) {
        auto sigmoid = std::make_shared<Sigmoid>();
        sTensor result = sigmoid->forward(a);
        result->gradFunction = sigmoid;
        return result;
    } else {
        return no_grad::sigmoid(a);
    }
}

sTensor Sigmoid::forwardFn(const sTensor& a) {
    sTensor result = no_grad::sigmoid(a);
    if (a->requiresGrad) {
        cacheA = result->copy();
    }
    return result;
}

std::vector<sTensor> Sigmoid::backwardFn(sTensor grad) {
    if (!parents[0]->requiresGrad) {
        return {nullptr};
    }

    sTensor ones = std::make_shared<Tensor>(cacheA->shape, cacheA->location);
    fill(1.0f, ones);

    sTensor gradA = no_grad::hadamard(grad, no_grad::hadamard(cacheA, no_grad::subtract(ones, cacheA)));
    return {gradA};
}
