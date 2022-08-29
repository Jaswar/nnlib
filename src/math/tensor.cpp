/**
 * @file tensor.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 26 August 2022
 */

#include <string>
#include <exceptions/size_mismatch_exception.h>
#include <utils/location_verifiers.h>
#include <exceptions/different_data_location_exception.h>
#include <exceptions/unsupported_operation_exception.h>
#include "tensor.h"
#include "../gpu/allocation_gpu.cuh"
#include "tensor_operations_on_host.h"
#include "tensor_operations_on_device.cuh"

Tensor::Tensor() : shape(), size(0), location(HOST), device(nullptr) {
    host = nullptr;
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
        host = copy1DArray(size, other.host);
    } else {
        device = copy1DArrayDevice(size, other.device);
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (&other == this) {
        return *this;
    }

    if (size > 0) {
        if (location == HOST) {
            free(host);
        } else {
            free1DArrayDevice(device);
        }
    }

    location = other.location;
    // This copies the vector
    shape = other.shape;
    size = other.size;

    if (size > 0) {
        if (location == HOST) {
            host = copy1DArray(size, other.host);
        } else {
            device = copy1DArrayDevice(size, other.device);
        }
    }

    return *this;
}

void Tensor::moveToDevice() {
    if (location == DEVICE) {
        return;
    }

    float* deviceData = allocate1DArrayDevice(size);
    copy1DFromHostToDevice(host, deviceData, size);

    free(host);
    device = deviceData;
    location = DEVICE;
}

void Tensor::moveToHost() {
    if (location == HOST) {
        return;
    }

    float* hostData = allocate1DArray(size);
    copy1DFromDeviceToHost(device, hostData, size);

    free(device);
    host = hostData;
    location = HOST;
}

Tensor::~Tensor() {
    if (size == 0) {
        return;
    }

    if (location == HOST) {
        free(host);
    } else {
        free1DArrayDevice(device);
    }
}

void Tensor::computeSize() {
    size = 1;
    for (auto it = shape.begin(); it < shape.end(); it++) {
        size *= *it;
    }
}

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

void multiplyMatrixVector(const Tensor& matrix, const Tensor& vector, Tensor& destination) {
    if (matrix.shape[1] != vector.shape[0] || matrix.shape[0] != destination.shape[0]) {
        throw SizeMismatchException();
    }

    std::initializer_list<DataLocation> locations = {matrix.location, vector.location, destination.location};
    if (allLocationsAreHost(locations)) {
        multiplyMatrixVectorOnHost(matrix, vector, destination);
    } else if (allLocationsAreDevice(locations)) {
        multiplyMatrixMatrixOnDevice(matrix, vector, destination);
    } else {
        throw DifferentDataLocationException();
    }
}

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
}


