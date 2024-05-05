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

sTensor sum(sTensor a) {
    auto sum = std::make_shared<SumReduce>();
    sTensor result = sum->forward({std::move(a)});
    result->gradFunction = sum;
    return result;
}

sTensor SumReduce::forwardFn(const std::vector<sTensor>& args) {
    shapeCache = args[0]->shape;
    DataLocation original = args[0]->location;
    args[0]->move(HOST);

    sTensor result = std::make_shared<Tensor>(1);
    float sum = sumTensor(*args[0]);
    result->data[0] = sum;

    result->move(original);
    args[0]->move(original);

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

sTensor add(sTensor a, sTensor b) {
    auto add = std::make_shared<Add>();
    sTensor result = add->forward({std::move(a), std::move(b)});
    result->gradFunction = add;
    return result;
}

sTensor Add::forwardFn(const std::vector<sTensor>& args) {
    if (args[0]->shape != args[1]->shape) {
        throw SizeMismatchException();
    }

    sTensor result = std::make_shared<Tensor>(args[0]->shape);
    result->move(args[0]->location);

    std::initializer_list<DataLocation> locations = {args[0]->location, args[1]->location};
    if (allLocationsAreHost(locations)) {
        addTensorsOnHost(*args[0], *args[1], *result);
    } else if (allLocationsAreDevice(locations)) {
        addTensorsOnDevice(*args[0], *args[1], *result);
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


sTensor subtract(sTensor a, sTensor b) {
    auto subtract = std::make_shared<Subtract>();
    sTensor result = subtract->forward({std::move(a), std::move(b)});
    result->gradFunction = subtract;
    return result;
}

sTensor Subtract::forwardFn(const std::vector<sTensor>& args) {
    if (args[0]->shape != args[1]->shape) {
        throw SizeMismatchException();
    }

    sTensor result = std::make_shared<Tensor>(args[0]->shape);
    result->move(args[0]->location);

    std::initializer_list<DataLocation> locations = {args[0]->location, args[1]->location};
    if (allLocationsAreHost(locations)) {
        subtractTensorsOnHost(*args[0], *args[1], *result);
    } else if (allLocationsAreDevice(locations)) {
        subtractTensorsOnDevice(*args[0], *args[1], *result);
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
    sTensor result = hadamard->forward({std::move(a), std::move(b)});
    result->gradFunction = hadamard;
    return result;
}

sTensor Hadamard::forwardFn(const std::vector<sTensor>& args) {
    if (args[0]->shape != args[1]->shape) {
        throw SizeMismatchException();
    }
    cacheA = args[0]->copy();
    cacheB = args[1]->copy();

    sTensor result = std::make_shared<Tensor>(args[0]->shape);
    result->move(args[0]->location);

    std::initializer_list<DataLocation> locations = {args[0]->location, args[1]->location};
    if (allLocationsAreHost(locations)) {
        hadamardTensorsOnHost(*args[0], *args[1], *result);
    } else if (allLocationsAreDevice(locations)) {
        hadamardTensorsOnDevice(*args[0], *args[1], *result);
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
    sTensor result = divide->forward({std::move(a), std::move(b)});
    result->gradFunction = divide;
    return result;
}

sTensor Divide::forwardFn(const std::vector<sTensor>& args) {
    if (args[0]->shape != args[1]->shape) {
        throw SizeMismatchException();
    }
    cacheA = args[0]->copy();
    cacheB = args[1]->copy();

    sTensor result = std::make_shared<Tensor>(args[0]->shape);
    result->move(args[0]->location);

    std::initializer_list<DataLocation> locations = {args[0]->location, args[1]->location};
    if (allLocationsAreHost(locations)) {
        divideTensorsOnHost(*args[0], *args[1], *result);
    } else if (allLocationsAreDevice(locations)) {
        divideTensorsOnDevice(*args[0], *args[1], *result);
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
    sTensor result = log->forward({std::move(a)});
    result->gradFunction = log;
    return result;
}

sTensor Log::forwardFn(const std::vector<sTensor>& args) {
    cacheA = args[0]->copy();
    sTensor result = std::make_shared<Tensor>(args[0]->shape);
    result->move(args[0]->location);

    std::initializer_list<DataLocation> locations = {args[0]->location};
    if (allLocationsAreHost(locations)) {
        logTensorOnHost(*args[0], *result);
    } else if (allLocationsAreDevice(locations)) {
        throw UnsupportedOperationException();
    } else {
        throw DifferentDataLocationException();
    }

    return result;
}

std::vector<sTensor> Log::backwardFn(sTensor grad) {
    sTensor gradA = divide(grad, cacheA);
    return {gradA};
}

sTensor matvecmul(sTensor a, sTensor b) {
    auto matvecmul = std::make_shared<MatVecMul>();
    sTensor result = matvecmul->forward({std::move(a), std::move(b)});
    result->gradFunction = matvecmul;
    return result;
}

sTensor matmul(sTensor a, sTensor b) {
    auto matmul = std::make_shared<Matmul>();
    sTensor result = matmul->forward({std::move(a), std::move(b)});
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

sTensor MatVecMul::forwardFn(const std::vector<sTensor>& args) {
    if (args[0]->shape[1] != args[1]->shape[0]) {
        throw SizeMismatchException();
    }
    cacheA = args[0]->copy();
    cacheB = args[1]->copy();
    sTensor result = std::make_shared<Tensor>(args[0]->shape[0]);
    result->move(args[0]->location);

    std::initializer_list<DataLocation> locations = {args[0]->location, args[1]->location};
    if (allLocationsAreHost(locations)) {
        multiplyMatrixVectorOnHost(*args[0], *args[1], *result);
    } else if (allLocationsAreDevice(locations)) {
        multiplyMatrixVectorOnDevice(*args[0], *args[1], *result);
    } else {
        throw DifferentDataLocationException();
    }

    return result;
}

std::vector<sTensor> MatVecMul::backwardFn(sTensor grad) {
    cacheB->shape = {cacheB->shape[0], 1};
    sTensor gradA = multiply(grad, transpose(cacheB));
    sTensor gradB = multiply(transpose(cacheA), grad);
    return {gradA, gradB};
}

sTensor Matmul::forwardFn(const std::vector<sTensor>& args) {
    if (args[0]->shape[1] != args[1]->shape[0]) {
        throw SizeMismatchException();
    }

    cacheA = args[0]->copy();
    cacheB = args[1]->copy();
    sTensor result = std::make_shared<Tensor>(args[0]->shape[0], args[1]->shape[1]);
    result->move(args[0]->location);

    std::initializer_list<DataLocation> locations = {args[0]->location, args[1]->location};
    if (allLocationsAreHost(locations)) {
        multiplyMatrixMatrixOnHost(*args[0], *args[1], *result);
    } else if (allLocationsAreDevice(locations)) {
        multiplyMatrixMatrixOnDevice(*args[0], *args[1], *result);
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
    sTensor result = transpose->forward({std::move(a)});
    result->gradFunction = transpose;
    return result;
}

sTensor Transpose::forwardFn(const std::vector<sTensor>& args) {
    cacheA = args[0]->copy();
    sTensor result = std::make_shared<Tensor>(args[0]->shape[1], args[0]->shape[0]);
    result->move(args[0]->location);

    std::initializer_list<DataLocation> locations = {args[0]->location};
    if (allLocationsAreHost(locations)) {
        transposeMatrixOnHost(*args[0], *result);
    } else if (allLocationsAreDevice(locations)) {
        transposeMatrixOnDevice(*args[0], *result);
    } else {
        throw DifferentDataLocationException();
    }

    return result;
}

std::vector<sTensor> Transpose::backwardFn(sTensor grad) {
    return {transpose(grad)};
}

