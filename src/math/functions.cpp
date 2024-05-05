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

void fill(float value, sTensor tensor) {
    fill(value, *tensor);
}

sTensor sum(sTensor a) {
    auto sum = std::make_shared<SumReduce>();
    sTensor result = sum->forward(a);
    result->gradFunction = sum;
    return result;
}

sTensor SumReduce::forwardFn(const sTensor& a) {
    shapeCache = a->shape;
    DataLocation original = a->location;
    a->move(HOST);

    sTensor result = std::make_shared<Tensor>(1);
    float sum = sumTensor(*a);
    result->data[0] = sum;

    result->move(original);
    a->move(original);

    return result;
}

std::vector<sTensor> SumReduce::backwardFn(sTensor grad) {
    DataLocation original = grad->location;
    grad->move(HOST);

    sTensor gradA = std::make_shared<Tensor>(shapeCache);
    gradA->move(grad->location);
    fill(grad->data[0], *gradA);

    grad->move(original);
    gradA->move(original);

    return {gradA};
}

sTensor addTensors(sTensor a, sTensor b) {
    auto add = std::make_shared<Add>();
    sTensor result = add->forward(a, b);
    result->gradFunction = add;
    return result;
}

sTensor addBroadcast(sTensor a, sTensor b) {
    auto add = std::make_shared<AddBroadcast>();
    sTensor result = add->forward(a, b);
    result->gradFunction = add;
    return result;
}

sTensor add(sTensor a, sTensor b) {
    if (a->shape.size() == 2 && b->shape.size() == 1) {
        return addBroadcast(a, b);
    } else {
        return addTensors(a, b);
    }
}

sTensor Add::forwardFn(const sTensor& a, const sTensor& b) {
    if (a->shape != b->shape) {
        throw SizeMismatchException();
    }

    sTensor result = std::make_shared<Tensor>(a->shape);
    result->move(a->location);

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

std::vector<sTensor> Add::backwardFn(sTensor grad) {
    sTensor gradA = grad->copy();
    sTensor gradB = grad->copy();
    return {gradA, gradB};
}

sTensor AddBroadcast::forwardFn(const sTensor& a, const sTensor& b) {
    if (a->shape[1] != b->shape[0]) {
        throw SizeMismatchException();
    }

    sTensor result = std::make_shared<Tensor>(a->shape[0], a->shape[1]);
    result->move(a->location);

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

std::vector<sTensor> AddBroadcast::backwardFn(sTensor grad) {
    sTensor gradA = grad->copy();
    sTensor ones = std::make_shared<Tensor>(grad->shape[0]);
    ones->move(grad->location);
    fill(1.0f, ones);
    sTensor gradB = multiply(transpose(grad), ones);  // TODO: replace later with sum reduction
    return {gradA, gradB};
}

sTensor subtract(sTensor a, sTensor b) {
    auto subtract = std::make_shared<Subtract>();
    sTensor result = subtract->forward(a, b);
    result->gradFunction = subtract;
    return result;
}

sTensor Subtract::forwardFn(const sTensor& a, const sTensor& b) {
    if (a->shape != b->shape) {
        throw SizeMismatchException();
    }

    sTensor result = std::make_shared<Tensor>(a->shape);
    result->move(a->location);

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

std::vector<sTensor> Subtract::backwardFn(sTensor grad) {
    sTensor gradA = grad->copy();
    sTensor gradB = grad->copy();
    gradB = multiply(gradB, -1.0f);
    return {gradA, gradB};
}


sTensor hadamard(sTensor a, sTensor b) {
    auto hadamard = std::make_shared<Hadamard>();
    sTensor result = hadamard->forward(a, b);
    result->gradFunction = hadamard;
    return result;
}

sTensor Hadamard::forwardFn(const sTensor& a, const sTensor& b) {
    if (a->shape != b->shape) {
        throw SizeMismatchException();
    }
    cacheA = a->copy();
    cacheB = b->copy();

    sTensor result = std::make_shared<Tensor>(a->shape);
    result->move(a->location);

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

std::vector<sTensor> Hadamard::backwardFn(sTensor grad) {
    sTensor gradA = hadamard(grad, cacheB);
    sTensor gradB = hadamard(grad, cacheA);
    return {gradA, gradB};
}


sTensor divide(sTensor a, sTensor b) {
    auto divide = std::make_shared<Divide>();
    sTensor result = divide->forward(a, b);
    result->gradFunction = divide;
    return result;
}

sTensor Divide::forwardFn(const sTensor& a, const sTensor& b) {
    if (a->shape != b->shape) {
        throw SizeMismatchException();
    }
    cacheA = a->copy();
    cacheB = b->copy();

    sTensor result = std::make_shared<Tensor>(a->shape);
    result->move(b->location);

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

std::vector<sTensor> Divide::backwardFn(sTensor grad) {
    sTensor gradA = divide(grad, cacheB);
    sTensor gradB = divide(hadamard(grad, cacheA), hadamard(cacheB, cacheB));
    gradB = multiply(gradB, -1.0f);
    return {gradA, gradB};
}

sTensor log(sTensor a) {
    auto log = std::make_shared<Log>();
    sTensor result = log->forward(a);
    result->gradFunction = log;
    return result;
}

sTensor Log::forwardFn(const sTensor& a) {
    cacheA = a->copy();
    sTensor result = std::make_shared<Tensor>(a->shape);
    result->move(a->location);

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

std::vector<sTensor> Log::backwardFn(sTensor grad) {
    sTensor gradA = divide(grad, cacheA);
    return {gradA};
}

sTensor multiply(sTensor a, float constant) {
    auto multiply = std::make_shared<MulConstant>();
    sTensor result = multiply->forward(a, constant);
    result->gradFunction = multiply;
    return result;
}

sTensor MulConstant::forwardFn(const sTensor& a, const float& b) {
    constantCache = b;
    sTensor result = std::make_shared<Tensor>(a->shape);
    result->move(a->location);

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

std::vector<sTensor> MulConstant::backwardFn(sTensor grad) {
    sTensor gradA = multiply(grad, constantCache);
    return {gradA};
}

sTensor matvecmul(sTensor a, sTensor b) {
    auto matvecmul = std::make_shared<MatVecMul>();
    sTensor result = matvecmul->forward(a, b);
    result->gradFunction = matvecmul;
    return result;
}

sTensor matmul(sTensor a, sTensor b) {
    auto matmul = std::make_shared<Matmul>();
    sTensor result = matmul->forward(a, b);
    result->gradFunction = matmul;
    return result;
}

sTensor multiply(sTensor a, sTensor b) {
    if (a->shape.size() == 2 && b->shape.size() == 2) {
        return matmul(a, b);
    } else if (a->shape.size() == 2 && b->shape.size() == 1) {
        return matvecmul(a, b);
    } else {
        throw UnsupportedOperationException();
    }
}

sTensor MatVecMul::forwardFn(const sTensor& a, const sTensor& b) {
    if (a->shape[1] != b->shape[0]) {
        throw SizeMismatchException();
    }

    cacheA = a->copy();
    cacheB = b->copy();
    sTensor result = std::make_shared<Tensor>(a->shape[0]);
    result->move(a->location);

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

std::vector<sTensor> MatVecMul::backwardFn(sTensor grad) {
    cacheB->shape = {cacheB->shape[0], 1};
    grad->shape = {grad->shape[0], 1};
    sTensor gradA = multiply(grad, transpose(cacheB));
    sTensor gradB = multiply(transpose(cacheA), grad);
    gradB->shape = {gradB->shape[0]};
    return {gradA, gradB};
}

sTensor Matmul::forwardFn(const sTensor& a, const sTensor& b) {
    if (a->shape[1] != b->shape[0]) {
        throw SizeMismatchException();
    }

    cacheA = a->copy();
    cacheB = b->copy();
    sTensor result = std::make_shared<Tensor>(a->shape[0], b->shape[1]);
    result->move(a->location);

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

std::vector<sTensor> Matmul::backwardFn(sTensor grad) {
    sTensor gradA = multiply(grad, transpose(cacheB));
    sTensor gradB = multiply(transpose(cacheA), grad);
    return {gradA, gradB};
}

sTensor transpose(sTensor a) {
    auto transpose = std::make_shared<Transpose>();
    sTensor result = transpose->forward(a);
    result->gradFunction = transpose;
    return result;
}

sTensor Transpose::forwardFn(const sTensor& a) {
    cacheA = a->copy();
    sTensor result = std::make_shared<Tensor>(a->shape[1], a->shape[0]);
    result->move(a->location);

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

std::vector<sTensor> Transpose::backwardFn(sTensor grad) {
    return {transpose(grad)};
}

