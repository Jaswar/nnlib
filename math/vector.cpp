//
// Created by Jan Warchocki on 03/03/2022.
//

#include "vector.h"
#include "../exceptions/size_mismatch_exception.h"
#include "../gpu/allocation_gpu.cuh"
#include "../exceptions/different_data_location_exception.h"
#include "../gpu/vector_operations.cuh"

Vector::Vector(int n) : Vector(allocate1DArray(n), n) {}

Vector::Vector(int n, dLocation location) : n(n), data(), location(location) {
    if (location == HOST) {
        this->data = allocate1DArray(n);
    } else {
        this->data = allocate1DArrayDevice(n);
    }
}

Vector::Vector(DTYPE* data, int n) : data(data), n(n), location(HOST) {}

Vector::Vector(DTYPE* data, int n, dLocation location) : data(data), n(n), location(location) {}

Vector::Vector(const Vector& vector) {
    location = vector.location;
    n = vector.n;

    if (location == HOST) {
        data = copy1DArray(n, vector.data);
    } else {
        data = copy1DArrayDevice(n, vector.data);
    }
}

Vector::~Vector() {
    if (location == HOST) {
        free(data);
    } else {
        free1DArrayDevice(data);
    }
}


void Vector::moveToDevice() {
    if (location == DEVICE) {
        return;
    }

    DTYPE* deviceData = allocate1DArrayDevice(n);
    copy1DFromHostToDevice(data, deviceData, n);

    free(data);
    data = deviceData;
    location = DEVICE;
}

void Vector::moveToHost() {
    if (location == HOST) {
        return;
    }

    DTYPE* hostData = allocate1DArray(n);
    copy1DFromDeviceToHost(data, hostData, n);

    free1DArrayDevice(data);
    data = hostData;
    location = HOST;
}

Vector& Vector::operator=(const Vector& vector) {
    n = vector.n;
    location = vector.location;

    if (location == HOST) {
        free(data);
        data = copy1DArray(n, vector.data);
    } else {
        free1DArrayDevice(data);
        data = copy1DArrayDevice(n, vector.data);
    }

    return *this;
}

DTYPE Vector::operator[](int index) const {
    return data[index];
}

// This can overwrite the value at index, so it is not const.
DTYPE& Vector::operator[](int index) {
    return data[index];
}

std::ostream& operator<<(std::ostream& stream, const Vector& vector) {
    int maxPeek = 10; // Number of lines to show from the front and back of vector.

    stream << "Vector([";

    for (int i = 0; i < vector.n - 1; i++) {
        if (i < maxPeek || i >= vector.n - maxPeek) {
            stream << vector[i] << ",\n\t";
        } else if (i == maxPeek) {
            stream << "[... " << vector.n - maxPeek * 2 << " more rows ...]\n\t";
        }
    }

    if (vector.n > 0) {
        stream << vector[vector.n - 1];
    }

    stream << "])";

    return stream;
}

Vector operator+(const Vector& v1, const Vector& v2) {
    if (v1.n != v2.n) {
        throw SizeMismatchException();
    }
    if (v1.location != v2.location) {
        throw DifferentDataLocationException();
    }

    if (v1.location == HOST) {
        DTYPE* newData = allocate1DArray(v1.n);

        for (int i = 0; i < v1.n; i++) {
            newData[i] = v1[i] + v2[i];
        }

        return Vector(newData, v1.n);
    } else {
        return addVectors(v1, v2);
    }
}

Vector operator-(const Vector& v1, const Vector& v2) {
    if (v1.n != v2.n) {
        throw SizeMismatchException();
    }
    if (v1.location != v2.location) {
        throw DifferentDataLocationException();
    }

    if (v1.location == HOST) {
        DTYPE* newData = allocate1DArray(v1.n);

        for (int i = 0; i < v1.n; i++) {
            newData[i] = v1[i] - v2[i];
        }

        return Vector(newData, v1.n);
    } else {
        return subtractVectors(v1, v2);
    }
}

Vector operator*(const Vector& v1, DTYPE constant) {
    if (v1.location == HOST) {
        DTYPE* newData = allocate1DArray(v1.n);

        for (int i = 0; i < v1.n; i++) {
            newData[i] = v1[i] * constant;
        }

        return Vector(newData, v1.n);
    } else {
        return multiplyVector(v1, constant);
    }
}

Vector operator*(DTYPE constant, const Vector& v1) {
    return v1 * constant;
}
