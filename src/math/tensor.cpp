/**
 * @file tensor.cpp
 * @brief Source file defining methods related to the Tensor class.
 * @author Jan Warchocki
 * @date 26 August 2022
 */

#include "tensor.h"
#include "../gpu/allocation_gpu.cuh"
#include "cache.h"
#include "tensor_operations_on_device.cuh"
#include "tensor_operations_on_host.h"
#include <exceptions/different_data_location_exception.h>
#include <exceptions/size_mismatch_exception.h>
#include <exceptions/unsupported_operation_exception.h>
#include <functions.h>
#include <memory>
#include <queue>
#include <string>
#include <utils/location_verifiers.h>

Tensor::Tensor() : shape(), size(0), location(HOST), data(), requiresGrad(false), gradFunction(), grad() {
}

Tensor::Tensor(std::vector<size_t> shape) : shape(std::move(shape)), location(HOST), size(0), data(), requiresGrad(false), gradFunction(), grad() {
    computeSize();
    Cache& cache = Cache::getInstance();
    data = cache.get(size, location);
}

Tensor::Tensor(std::vector<size_t> shape, DataLocation location) : shape(std::move(shape)), location(location), size(0), data(), requiresGrad(false), gradFunction(), grad() {
    computeSize();
    Cache& cache = Cache::getInstance();
    data = cache.get(size, location);
}

Tensor::Tensor(const Tensor& other) {
    location = other.location;
    // This copies the vector
    shape = other.shape;
    size = other.size;
    requiresGrad = other.requiresGrad;
    gradFunction = other.gradFunction;
    if (other.grad != nullptr) {
        grad = std::make_shared<Tensor>(*other.grad);
    }

    if (size == 0) {
        return;
    }

    Cache& cache = Cache::getInstance();
    data = cache.get(size, other.location);
    if (location == HOST) {
        copy1DArray(size, other.data, data);
    } else {
        copy1DArrayDevice(size, other.data, data);
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (&other == this) {
        return *this;
    }

    Cache& cache = Cache::getInstance();
    cache.put(size, data, location);  // Mark the memory as reusable

    location = other.location;
    // This copies the vector
    shape = other.shape;
    size = other.size;

    data = cache.get(size, other.location);
    if (location == HOST) {
        copy1DArray(size, other.data, data);
    } else {
        copy1DArrayDevice(size, other.data, data);
    }

    return *this;
}

void Tensor::move(DataLocation target) {
    if (location == target) {
        return;
    }

    Cache& cache = Cache::getInstance();
    cache.put(size, data, location);  // Mark the memory as reusable
    float* newData = cache.get(size, target);
    if (location == HOST) {
        copy1DFromHostToDevice(data, newData, size);
    } else {
        copy1DFromDeviceToHost(data, newData, size);
    }
    data = newData;
    location = target;
    if (grad != nullptr) {
        grad->move(target);
    }
}

Tensor::~Tensor() {
    if (size == 0) {
        return;
    }
    Cache& cache = Cache::getInstance();
    cache.put(size, data, location);
}

void Tensor::computeSize() {
    size = 1;
    for (auto it = shape.begin(); it < shape.end(); it++) {
        size *= *it;
    }
}

Tensor Tensor::construct1d(const std::vector<float>& data) {
    if (data.empty()) {
        throw SizeMismatchException();
    }
    Tensor result = Tensor(data.size());
    std::copy(data.begin(), data.end(), result.data);
    return result;
}

Tensor Tensor::construct2d(const std::vector<std::vector<float>>& data) {
    if (data.empty() || data[0].empty()) {
        throw SizeMismatchException();
    }

    Tensor result = Tensor(data.size(), data[0].size());
    for (size_t i = 0; i < data.size(); i++) {
        // Make sure the array has the same number of columns in each row
        if (data[i].size() != result.shape[1]) {
            throw SizeMismatchException();
        }

        for (size_t j = 0; j < data[0].size(); j++) {
            result.data[i * result.shape[1] + j] = data[i][j];
        }
    }
    return result;
}

size_t Tensor::findEffectiveAddress(const std::vector<size_t>& index, size_t depth) const {
    if (depth == 0) {
        return index.front();
    }

    return shape.at(depth) * findEffectiveAddress(index, depth - 1) + index.at(depth);
}

void Tensor::verifyIndex(const std::vector<size_t>& index) const {
    if (index.size() != shape.size()) {
        throw SizeMismatchException();
    }
    for (size_t i = 0; i < index.size(); i++) {
        if (index[i] >= shape[i]) {
            throw SizeMismatchException();
        }
    }
}

void Tensor::useGrad() {
    requiresGrad = true;
    grad = std::make_shared<Tensor>(shape, location);
    fill(0.0f, grad);
    gradFunction = nullptr;
}

void Tensor::backward() {
    if (shape.size() != 1 || shape[0] != 1 || !requiresGrad) {
        throw UnsupportedOperationException();
    }

    sTensor gradient = std::make_shared<Tensor>(shape);
    fill(1.0f, gradient);
    gradient->move(location);
    sTensor current = std::make_shared<Tensor>(*this);

    std::queue<std::pair<sTensor, sTensor>> queue;
    queue.emplace(current, gradient);
    while (!queue.empty()) {
        current = queue.front().first;
        gradient = queue.front().second;
        queue.pop();

        std::shared_ptr<BackwardFunction> gradFn = current->gradFunction;
        if (gradFn == nullptr) {
            if (current->requiresGrad) {
                current->grad = no_grad::add(current->grad, gradient);
            }
            continue;
        }

        std::vector<sTensor> newGrads = gradFn->backward(gradient);
        for (int i = 0; i < newGrads.size(); i++) {
            sTensor parent = gradFn->parents[i];
            if (parent->requiresGrad) {
                queue.emplace(parent, newGrads[i]);
            }
        }
    }
}

std::shared_ptr<Tensor> Tensor::copy() const {
    sTensor copy = std::make_shared<Tensor>(shape, location);
    copy->requiresGrad = requiresGrad;
    copy->grad = nullptr;
    if (grad != nullptr) {
        copy->grad = grad->copy();
    }
    copy->gradFunction = gradFunction;

    if (location == HOST) {
        copy1DArray(size, data, copy->data);
    } else {
        copy1DArrayDevice(size, data, copy->data);
    }

    return copy;
}

/**
 * @brief Convert the shape of the tensor to a string.
 *
 * The shape is displayed in the format: `[2, 2, 3]`.
 *
 * @param tensor The tensor whose shape to show as a string.
 * @return The string representation of the shape of the tensor.
 */
std::string tensorShapeToString(const Tensor& tensor) {
    std::string shapeString = "[";

    for (auto it = tensor.shape.begin(); it < tensor.shape.end(); it++) {
        shapeString += std::to_string(*it);
        if (it != tensor.shape.end() - 1) {
            shapeString += ", ";
        }
    }

    return shapeString + "]";
}

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor) {
    stream << "Tensor: ";
    for (int i = 0; i < tensor.size; i++) {
        stream << tensor.data[i] << " ";
    }
    return stream;
//    if (tensor.location == DEVICE) {
//        return stream << "Tensor located on device with shape: " + tensorShapeToString(tensor);
//    } else {
//        return stream << "Tensor located on host with shape: " + tensorShapeToString(tensor);
//    }
}

float sum(Tensor& tensor) {
    const DataLocation& oldLocation = tensor.location;
    tensor.move(HOST);
    float sum = sumTensor(tensor);
    tensor.move(oldLocation);
    return sum;
}

void fill(float value, Tensor& destination) {
    if (destination.location == HOST) {
        fillTensorOnHost(destination, value);
    } else {
        fillTensorOnDevice(destination, value);
    }
}

/**
 * @brief Method to perform element-wise addition on tensors.
 *
 * @param a The first tensor.
 * @param b The second tensor.
 * @param destination Where the result of the addition should be stored.
 */
void addTensors(const Tensor& a, const Tensor& b, Tensor& destination) {
    if (a.shape != b.shape || a.shape != destination.shape || b.shape != destination.shape) {
        throw SizeMismatchException();
    }

    std::initializer_list<DataLocation> locations = {a.location, b.location, destination.location};
    if (allLocationsAreHost(locations)) {
        addTensorsOnHost(a, b, destination);
    } else if (allLocationsAreDevice(locations)) {
        addTensorsOnDevice(a, b, destination);
    } else {
        throw DifferentDataLocationException();
    }
}

/**
 * @brief Method to perform broadcast-add operation on tensors.
 *
 * This operation adds @p vector to every row of @p matrix.
 *
 * @param matrix The first tensor, must be a matrix.
 * @param vector The second tensor, must be a vector.
 * @param destination Where the result of the addition should be stored.
 */
void addBroadcast(const Tensor& matrix, const Tensor& vector, Tensor& destination) {
    if (matrix.shape[1] != vector.shape[0] || matrix.shape != destination.shape) {
        throw SizeMismatchException();
    }

    std::initializer_list<DataLocation> locations = {matrix.location, vector.location, destination.location};
    if (allLocationsAreHost(locations)) {
        addBroadcastOnHost(matrix, vector, destination);
    } else if (allLocationsAreDevice(locations)) {
        addBroadcastOnDevice(matrix, vector, destination);
    } else {
        throw DifferentDataLocationException();
    }
}


void add(const Tensor& a, const Tensor& b, Tensor& destination) {
    if (a.shape.size() == 2 && b.shape.size() == 1 && destination.shape.size() == 2) {
        addBroadcast(a, b, destination);
    } else {
        addTensors(a, b, destination);
    }
}

Tensor add(const Tensor& a, const Tensor& b) {
    Tensor result = Tensor(a.shape);
    result.move(a.location);
    add(a, b, result);
    return result;
}

void subtract(const Tensor& a, const Tensor& b, Tensor& destination) {
    if (a.shape != b.shape || a.shape != destination.shape || b.shape != destination.shape) {
        throw SizeMismatchException();
    }

    std::initializer_list<DataLocation> locations = {a.location, b.location, destination.location};
    if (allLocationsAreHost(locations)) {
        subtractTensorsOnHost(a, b, destination);
    } else if (allLocationsAreDevice(locations)) {
        subtractTensorsOnDevice(a, b, destination);
    } else {
        throw DifferentDataLocationException();
    }
}

Tensor subtract(const Tensor& a, const Tensor& b) {
    Tensor result = Tensor(a.shape);
    result.move(a.location);
    subtract(a, b, result);
    return result;
}

Tensor hadamard(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) {
        throw SizeMismatchException();
    }

    Tensor result = Tensor(a.shape);
    result.move(a.location);

    std::initializer_list<DataLocation> locations = {a.location, b.location};
    if (allLocationsAreHost(locations)) {
        hadamardTensorsOnHost(a, b, result);
    } else if (allLocationsAreDevice(locations)) {
        hadamardTensorsOnDevice(a, b, result);
    } else {
        throw DifferentDataLocationException();
    }
    return result;
}

Tensor divide(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) {
        throw SizeMismatchException();
    }

    Tensor result = Tensor(a.shape);
    result.move(a.location);

    std::initializer_list<DataLocation> locations = {a.location, b.location};
    if (allLocationsAreHost(locations)) {
        divideTensorsOnHost(a, b, result);
    } else if (allLocationsAreDevice(locations)) {
        divideTensorsOnDevice(a, b, result);
    } else {
        throw DifferentDataLocationException();
    }
    return result;
}

Tensor log(const Tensor& a) {
    Tensor result = Tensor(a.shape);
    result.move(a.location);

    std::initializer_list<DataLocation> locations = {a.location};
    if (allLocationsAreHost(locations)) {
        logTensorOnHost(a, result);
    } else if (allLocationsAreDevice(locations)) {
        logTensorOnDevice(a, result);
    } else {
        throw DifferentDataLocationException();
    }
    return result;
}

Tensor multiply(const Tensor& tensor, float constant) {
    Tensor result = Tensor(tensor.shape);
    result.move(tensor.location);

    std::initializer_list<DataLocation> locations = {tensor.location};
    if (allLocationsAreHost(locations)) {
        multiplyTensorOnHost(tensor, constant, result);
    } else if (allLocationsAreDevice(locations)) {
        multiplyTensorOnDevice(tensor, constant, result);
    } else {
        throw DifferentDataLocationException();
    }
    return result;
}

/**
 * @brief Method to perform matrix-vector multiplication on tensors.
 *
 * @param matrix The matrix tensor.
 * @param vector The vector tensor.
 * @param destination Where the result of the multiplication should be stored.
 */
Tensor multiplyMatrixVector(const Tensor& matrix, const Tensor& vector) {
    if (matrix.shape[1] != vector.shape[0]) {
        throw SizeMismatchException();
    }

    Tensor result = Tensor(matrix.shape[0]);
    result.move(matrix.location);

    std::initializer_list<DataLocation> locations = {matrix.location, vector.location};
    if (allLocationsAreHost(locations)) {
        multiplyMatrixVectorOnHost(matrix, vector, result);
    } else if (allLocationsAreDevice(locations)) {
        multiplyMatrixVectorOnDevice(matrix, vector, result);
    } else {
        throw DifferentDataLocationException();
    }
    return result;
}

/**
 * @brief Method to perform matrix-matrix multiplication on tensors.
 *
 * @param m1 The first matrix tensor.
 * @param m2 The second matrix tensor.
 * @param destination Where the result of the multiplication should be stored.
 */
Tensor multiplyMatrixMatrix(const Tensor& m1, const Tensor& m2) {
    if (m1.shape[1] != m2.shape[0]) {
        throw SizeMismatchException();
    }
    Tensor result = Tensor(m1.shape[0], m2.shape[1]);
    result.move(m1.location);

    std::initializer_list<DataLocation> locations = {m1.location, m2.location};
    if (allLocationsAreHost(locations)) {
        multiplyMatrixMatrixOnHost(m1, m2, result);
    } else if (allLocationsAreDevice(locations)) {
        multiplyMatrixMatrixOnDevice(m1, m2, result);
    } else {
        throw DifferentDataLocationException();
    }
    return result;
}

Tensor multiply(const Tensor& a, const Tensor& b) {
    if (a.shape.size() == 2 && b.shape.size() == 1) {
        return multiplyMatrixVector(a, b);
    } else if (a.shape.size() == 2 && b.shape.size() == 2) {
        return multiplyMatrixMatrix(a, b);
    } else {
        throw UnsupportedOperationException();
    }
}

Tensor transpose(const Tensor& matrix) {
    Tensor result = Tensor(matrix.shape[1], matrix.shape[0]);
    result.move(matrix.location);

    std::initializer_list<DataLocation> locations = {matrix.location};
    if (allLocationsAreHost(locations)) {
        transposeMatrixOnHost(matrix, result);
    } else if (allLocationsAreDevice(locations)) {
        transposeMatrixOnDevice(matrix, result);
    } else {
        throw DifferentDataLocationException();
    }
    return result;
}

