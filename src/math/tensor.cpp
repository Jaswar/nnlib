/**
 * @file tensor.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 26 August 2022
 */

#include <string>
#include <exceptions/size_mismatch_exception.h>
#include <utils/location_verifiers.h>
#include "tensor.h"
#include "../gpu/allocation_gpu.cuh"

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


void add(const Tensor& a, const Tensor& b, Tensor& destination) {
    if (a.shape != b.shape || a.shape != destination.shape || b.shape != destination.shape) {
        throw SizeMismatchException();
    }

    std::initializer_list<DataLocation> locations = {a.location, b.location, destination.location};
    if (allLocationsAreHost(locations)) {

    } else {

    }
}


