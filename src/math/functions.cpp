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

sTensor multiply(sTensor a, sTensor b) {
    auto matmul = std::make_shared<Matmul>();
    sTensor result = matmul->forward({std::move(a), std::move(b)});
    result->gradFunction = matmul;
    return result;
}

sTensor Matmul::forwardFn(const std::vector<sTensor>& args) {
    if (args.size() != 2) {
        throw UnsupportedOperationException();
    }
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
    sTensor gradA = std::make_shared<Tensor>(multiply(*grad, transpose(*cacheB)));
    sTensor gradB = std::make_shared<Tensor>(multiply(transpose(*cacheA), *grad));
    return {gradA, gradB};
}