/**
 * @file tensor.cpp
 * @brief Source file defining methods related to the Tensor class.
 * @author Jan Warchocki
 * @date 26 August 2022
 */

#include "tensor.h"
#include "../gpu/allocation_gpu.cuh"
#include "tensor_operations_on_device.cuh"
#include "tensor_operations_on_host.h"
#include <exceptions/different_data_location_exception.h>
#include <exceptions/size_mismatch_exception.h>
#include <exceptions/unsupported_operation_exception.h>
#include <string>
#include <utils/location_verifiers.h>

Tensor::Tensor() : shape(), size(0), location(HOST), data() {
}

Tensor::Tensor(std::vector<size_t> shape) : shape(std::move(shape)), location(HOST), size(0), data() {
    computeSize();
    data = allocate1DArray(size, 0);
}

Tensor::Tensor(const Tensor& other) {
    location = other.location;
    // This copies the vector
    shape = other.shape;
    size = other.size;

    if (size == 0) {
        return;
    }

    if (location == HOST) {
        data = copy1DArray(size, other.data);
    } else {
        data = copy1DArrayDevice(size, other.data);
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (&other == this) {
        return *this;
    }

    if (size > 0) {
        if (location == HOST) {
            free(data);
        } else {
            free1DArrayDevice(data);
        }
    }

    location = other.location;
    // This copies the vector
    shape = other.shape;
    size = other.size;

    if (size > 0) {
        if (location == HOST) {
            data = copy1DArray(size, other.data);
        } else {
            data = copy1DArrayDevice(size, other.data);
        }
    }

    return *this;
}

void Tensor::move(DataLocation target) {
    if (location == target) {
        return;
    }

    if (location == HOST) {
        float* newData = allocate1DArrayDevice(size);
        copy1DFromHostToDevice(data, newData, size);
        free(data);
        data = newData;
    } else {
        float* newData = allocate1DArray(size);
        copy1DFromDeviceToHost(data, newData, size);
        free1DArrayDevice(data);
        data = newData;
    }
    location = target;
}

Tensor::~Tensor() {
    if (size == 0) {
        return;
    }

    if (location == HOST) {
        free(data);
    } else {
        free1DArrayDevice(data);
    }
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
    if (tensor.location == DEVICE) {
        return stream << "Tensor located on device with shape: " + tensorShapeToString(tensor);
    } else {
        return stream << "Tensor located on host with shape: " + tensorShapeToString(tensor);
    }
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

void hadamard(const Tensor& a, const Tensor& b, Tensor& destination) {
    if (a.shape != b.shape || a.shape != destination.shape || b.shape != destination.shape) {
        throw SizeMismatchException();
    }

    std::initializer_list<DataLocation> locations = {a.location, b.location, destination.location};
    if (allLocationsAreHost(locations)) {
        hadamardTensorsOnHost(a, b, destination);
    } else if (allLocationsAreDevice(locations)) {
        hadamardTensorsOnDevice(a, b, destination);
    } else {
        throw DifferentDataLocationException();
    }
}

void divide(const Tensor& a, const Tensor& b, Tensor& destination) {
    if (a.shape != b.shape || a.shape != destination.shape || b.shape != destination.shape) {
        throw SizeMismatchException();
    }

    std::initializer_list<DataLocation> locations = {a.location, b.location, destination.location};
    if (allLocationsAreHost(locations)) {
        divideTensorsOnHost(a, b, destination);
    } else if (allLocationsAreDevice(locations)) {
        divideTensorsOnDevice(a, b, destination);
    } else {
        throw DifferentDataLocationException();
    }
}

void log(const Tensor& a, Tensor& destination) {
    if (a.shape != destination.shape) {
        throw SizeMismatchException();
    }

    std::initializer_list<DataLocation> locations = {a.location, destination.location};
    if (allLocationsAreHost(locations)) {
        logTensorOnHost(a, destination);
    } else if (allLocationsAreDevice(locations)) {
        logTensorOnDevice(a, destination);
    } else {
        throw DifferentDataLocationException();
    }
}

void multiply(const Tensor& tensor, float constant, Tensor& destination) {
    if (tensor.shape != destination.shape) {
        throw SizeMismatchException();
    }

    std::initializer_list<DataLocation> locations = {tensor.location, destination.location};
    if (allLocationsAreHost(locations)) {
        multiplyTensorOnHost(tensor, constant, destination);
    } else if (allLocationsAreDevice(locations)) {
        multiplyTensorOnDevice(tensor, constant, destination);
    } else {
        throw DifferentDataLocationException();
    }
}

/**
 * @brief Method to perform matrix-vector multiplication on tensors.
 *
 * @param matrix The matrix tensor.
 * @param vector The vector tensor.
 * @param destination Where the result of the multiplication should be stored.
 */
void multiplyMatrixVector(const Tensor& matrix, const Tensor& vector, Tensor& destination) {
    if (matrix.shape[1] != vector.shape[0] || matrix.shape[0] != destination.shape[0]) {
        throw SizeMismatchException();
    }

    std::initializer_list<DataLocation> locations = {matrix.location, vector.location, destination.location};
    if (allLocationsAreHost(locations)) {
        multiplyMatrixVectorOnHost(matrix, vector, destination);
    } else if (allLocationsAreDevice(locations)) {
        multiplyMatrixVectorOnDevice(matrix, vector, destination);
    } else {
        throw DifferentDataLocationException();
    }
}

/**
 * @brief Method to perform matrix-matrix multiplication on tensors.
 *
 * @param m1 The first matrix tensor.
 * @param m2 The second matrix tensor.
 * @param destination Where the result of the multiplication should be stored.
 */
void multiplyMatrixMatrix(const Tensor& m1, const Tensor& m2, Tensor& destination) {
    if (m1.shape[1] != m2.shape[0] || m1.shape[0] != destination.shape[0] || m2.shape[1] != destination.shape[1]) {
        throw SizeMismatchException();
    }

    std::initializer_list<DataLocation> locations = {m1.location, m2.location, destination.location};
    if (allLocationsAreHost(locations)) {
        multiplyMatrixMatrixOnHost(m1, m2, destination);
    } else if (allLocationsAreDevice(locations)) {
        multiplyMatrixMatrixOnDevice(m1, m2, destination);
    } else {
        throw DifferentDataLocationException();
    }
}

void multiply(const Tensor& a, const Tensor& b, Tensor& destination) {
    if (a.shape.size() == 2 && b.shape.size() == 1 && destination.shape.size() == 1) {
        multiplyMatrixVector(a, b, destination);
    } else if (a.shape.size() == 2 && b.shape.size() == 2 && destination.shape.size() == 2) {
        multiplyMatrixMatrix(a, b, destination);
    } else {
        throw UnsupportedOperationException();
    }
}

void transpose(const Tensor& matrix, Tensor& destination) {
    if (matrix.shape.size() != 2 || destination.shape.size() != 2) {
        throw UnsupportedOperationException();
    }
    if (matrix.shape[0] != destination.shape[1] || matrix.shape[1] != destination.shape[0]) {
        throw SizeMismatchException();
    }

    std::initializer_list<DataLocation> locations = {matrix.location, destination.location};
    if (allLocationsAreHost(locations)) {
        transposeMatrixOnHost(matrix, destination);
    } else if (allLocationsAreDevice(locations)) {
        transposeMatrixOnDevice(matrix, destination);
    } else {
        throw DifferentDataLocationException();
    }
}
