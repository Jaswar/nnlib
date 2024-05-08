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
#include "tensor_operations_on_device.cuh"
#include "tensor_operations_on_host.h"
#include "utils/location_verifiers.h"

void fill(float value, sTensor& tensor) {
    fill(value, *tensor);
}

sTensor no_grad::sum(const sTensor& a) {
    DataLocation original = a->location;
    a->move(HOST);

    sTensor result = std::make_shared<Tensor>(1);
    float sum = sumTensor(*a);
    result->data[0] = sum;

    result->move(original);
    a->move(original);

    return result;
}

sTensor sum(const sTensor& a) {
    auto sum = std::make_shared<SumReduce>();
    sTensor result = sum->forward(a);
    result->gradFunction = sum;
    return result;
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

    DataLocation original = grad->location;
    grad->move(HOST);

    sTensor gradA = std::make_shared<Tensor>(shapeCache, grad->location);
    fill(grad->data[0], *gradA);

    grad->move(original);
    gradA->move(original);

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
    auto add = std::make_shared<Add>();
    sTensor result = add->forward(a, b);
    result->gradFunction = add;
    return result;
}

sTensor addBroadcast(const sTensor& a, const sTensor& b) {
    auto add = std::make_shared<AddBroadcast>();
    sTensor result = add->forward(a, b);
    result->gradFunction = add;
    return result;
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
    auto subtract = std::make_shared<Subtract>();
    sTensor result = subtract->forward(a, b);
    result->gradFunction = subtract;
    return result;
}

sTensor Subtract::forwardFn(const sTensor& a, const sTensor& b) {
    return no_grad::subtract(a, b);
}

std::vector<sTensor> Subtract::backwardFn(sTensor grad) {
    sTensor gradA = parents[0]->requiresGrad ? grad->copy() : nullptr;

    sTensor gradB = nullptr;
    if (parents[1]->requiresGrad) {
        gradB = grad->copy();
        gradB = no_grad::multiply(gradB, -1.0f);
    }
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
    auto hadamard = std::make_shared<Hadamard>();
    sTensor result = hadamard->forward(a, b);
    result->gradFunction = hadamard;
    return result;
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
    auto divide = std::make_shared<Divide>();
    sTensor result = divide->forward(a, b);
    result->gradFunction = divide;
    return result;
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
    auto log = std::make_shared<Log>();
    sTensor result = log->forward(a);
    result->gradFunction = log;
    return result;
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
    auto multiply = std::make_shared<MulConstant>();
    sTensor result = multiply->forward(a, constant);
    result->gradFunction = multiply;
    return result;
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
    auto matvecmul = std::make_shared<MatVecMul>();
    sTensor result = matvecmul->forward(a, b);
    result->gradFunction = matvecmul;
    return result;
}

sTensor matmul(const sTensor& a, const sTensor& b) {
    auto matmul = std::make_shared<Matmul>();
    sTensor result = matmul->forward(a, b);
    result->gradFunction = matmul;
    return result;
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
    auto transpose = std::make_shared<Transpose>();
    sTensor result = transpose->forward(a);
    result->gradFunction = transpose;
    return result;
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
    auto relu = std::make_shared<ReLU>();
    sTensor result = relu->forward(a);
    result->gradFunction = relu;
    return result;
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
    auto sigmoid = std::make_shared<Sigmoid>();
    sTensor result = sigmoid->forward(a);
    result->gradFunction = sigmoid;
    return result;
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
