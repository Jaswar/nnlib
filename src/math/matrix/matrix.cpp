/**
 * @file matrix.cpp
 * @brief Source file defining the Matrix class and operations on matrices.
 * @author Jan Warchocki
 * @date 03 March 2022
 */

#include "matrix.h"
#include "allocation.h"
#include "exceptions/different_data_location_exception.h"
#include "exceptions/size_mismatch_exception.h"
#include "gpu/allocation_gpu.cuh"
#include "matrix_operations_on_device.cuh"
#include "matrix_operations_on_host.h"
#include <utils/location_verifiers.h>

Matrix::Matrix(size_t n, size_t m) : Matrix(n, m, HOST) {
}

Matrix::Matrix(size_t n, size_t m, DataLocation location) : n(n), m(m), location(location) {
    if (location == HOST) {
        if (n > 0 && m > 0) {
            this->data = allocate1DArray(n * m);
        } else {
            this->data = nullptr;
        }
    } else {
        if (n > 0 && m > 0) {
            this->data = allocate1DArrayDevice(n * m);
        } else {
            this->data = nullptr;
        }
    }
}

Matrix::Matrix(DTYPE* data, size_t n, size_t m) : Matrix(data, n, m, HOST) {
}

Matrix::Matrix(DTYPE* data, size_t n, size_t m, DataLocation location) : data(data), n(n), m(m), location(location) {
}

Matrix::Matrix(const Matrix& matrix) {
    location = matrix.location;
    n = matrix.n;
    m = matrix.m;

    if (location == HOST) {
        data = copy1DArray(n * m, matrix.data);
    } else {
        data = copy1DArrayDevice(n * m, matrix.data);
    }
}

void Matrix::moveToDevice() {
    if (location == DEVICE) {
        return;
    }

    DTYPE* deviceData = allocate1DArrayDevice(n * m);
    copy1DFromHostToDevice(data, deviceData, n * m);

    free(data);
    data = deviceData;
    location = DEVICE;
}

void Matrix::moveToHost() {
    if (location == HOST) {
        return;
    }

    DTYPE* hostData = allocate1DArray(n * m);
    copy1DFromDeviceToHost(data, hostData, n * m);

    free1DArrayDevice(data);
    data = hostData;
    location = HOST;
}

Matrix::~Matrix() {
    if (n == 0 || m == 0) {
        return;
    }

    if (location == HOST) {
        free(data);
    } else {
        free1DArrayDevice(data);
    }
}

Matrix& Matrix::operator=(const Matrix& matrix) {
    if (this == &matrix) {
        return *this;
    }

    location = matrix.location;

    if (location == HOST) {
        if (n > 0 && m > 0) {
            free(data);
        }
        data = copy1DArray(matrix.n * matrix.m, matrix.data);
    } else {
        if (n > 0 && m > 0) {
            free1DArrayDevice(data);
        }
        data = copy1DArrayDevice(matrix.n * matrix.m, matrix.data);
    }

    n = matrix.n;
    m = matrix.m;

    return *this;
}

DTYPE& Matrix::operator()(size_t x, size_t y) const {
    return data[x * m + y];
}

void displayRow(std::ostream& stream, const Matrix& matrix, size_t row, size_t m) {
    stream << "[";

    for (int i = 0; i < m - 1; i++) {
        stream << matrix(row, i) << ", ";
    }

    if (m > 0) {
        stream << matrix(row, m - 1);
    }

    stream << "]";
}

std::ostream& operator<<(std::ostream& stream, const Matrix& matrix) {
    size_t maxPeek = 10; // Number of lines to show from the front and back of matrix.

    stream << "Matrix([";

    for (int i = 0; i < matrix.n - 1; i++) {
        // Make sure no more than 50 first and 50 last rows are displayed
        // to avoid flooding the console.
        if (i < maxPeek || i >= matrix.n - maxPeek) {
            displayRow(stream, matrix, i, matrix.m);

            stream << ",\n";
            stream << "\t";
        } else if (i == maxPeek) {
            stream << "[... " << matrix.n - maxPeek * 2 << " more rows ...]\n\t";
        }
    }

    if (matrix.n > 0) {
        displayRow(stream, matrix, matrix.n - 1, matrix.m);
    }

    stream << "])";

    return stream;
}

void add(const Matrix& m1, const Matrix& m2, Matrix& result) {
    // clang-format off
    if (m1.n != m2.n || m1.m != m2.m
        || m1.n != result.n || m1.m != result.m
        || m2.n != result.n || m2.m != result.m ) {
        throw SizeMismatchException();
    }
    // clang-format on

    const std::initializer_list<DataLocation> locations = {m1.location, m2.location, result.location};
    if (allLocationsAreHost(locations)) {
        addMatricesOnHost(m1, m2, result);
    } else if (allLocationsAreDevice(locations)) {
        addMatricesOnDevice(m1, m2, result);
    } else {
        throw DifferentDataLocationException();
    }
}

void add(const Matrix& m, const Vector& v, Matrix& result) {
    if (m.m != v.n || m.m != result.m || m.n != result.n || result.m != v.n) {
        throw SizeMismatchException();
    }

    const std::initializer_list<DataLocation> locations = {m.location, v.location, result.location};
    if (allLocationsAreHost(locations)) {
        addBroadcastOnHost(m, v, result);
    } else if (allLocationsAreDevice(locations)) {
        addBroadcastOnDevice(m, v, result);
    } else {
        throw DifferentDataLocationException();
    }
}

void subtract(const Matrix& m1, const Matrix& m2, Matrix& result) {
    // clang-format off
    if (m1.n != m2.n || m1.m != m2.m
        || m1.n != result.n || m1.m != result.m
        || m2.n != result.n || m2.m != result.m ) {
        throw SizeMismatchException();
    }
    // clang-format on

    const std::initializer_list<DataLocation> locations = {m1.location, m2.location, result.location};
    if (allLocationsAreHost(locations)) {
        subtractMatricesOnHost(m1, m2, result);
    } else if (allLocationsAreDevice(locations)) {
        subtractMatricesOnDevice(m1, m2, result);
    } else {
        throw DifferentDataLocationException();
    }
}

void multiply(const Matrix& m1, const Matrix& m2, Matrix& result) {
    if (m1.m != m2.n || m1.n != result.n || m2.m != result.m) {
        throw SizeMismatchException();
    }

    const std::initializer_list<DataLocation> locations = {m1.location, m2.location, result.location};
    if (allLocationsAreHost(locations)) {
        multiplyMatricesOnHost(m1, m2, result);
    } else if (allLocationsAreDevice(locations)) {
        multiplyMatricesOnDevice(m1, m2, result);
    } else {
        throw DifferentDataLocationException();
    }
}

void multiply(const Matrix& m, const Vector& v, Vector& result) {
    if (m.m != v.n || result.n != m.n) {
        throw SizeMismatchException();
    }

    const std::initializer_list<DataLocation> locations = {m.location, v.location, result.location};
    if (allLocationsAreHost(locations)) {
        multiplyMatrixVectorOnHost(m, v, result);
    } else if (allLocationsAreDevice(locations)) {
        multiplyMatrixVectorOnDevice(m, v, result);
    } else {
        throw DifferentDataLocationException();
    }
}

void multiply(const Matrix& m, DTYPE constant, Matrix& result) {
    if (m.n != result.n || m.m != result.m) {
        throw SizeMismatchException();
    }

    const std::initializer_list<DataLocation> locations = {m.location, result.location};
    if (allLocationsAreHost(locations)) {
        multiplyMatrixOnHost(m, constant, result);
    } else if (allLocationsAreDevice(locations)) {
        multiplyMatrixOnDevice(m, constant, result);
    } else {
        throw DifferentDataLocationException();
    }
}

void multiply(DTYPE constant, const Matrix& m, Matrix& result) {
    multiply(m, constant, result);
}

void hadamard(const Matrix& m1, const Matrix& m2, Matrix& result) {
    // clang-format off
    if (m1.n != m2.n || m1.m != m2.m
        || m1.n != result.n || m1.m != result.m
        || m2.n != result.n || m2.m != result.m ) {
        throw SizeMismatchException();
    }
    // clang-format on

    const std::initializer_list<DataLocation> locations = {m1.location, m2.location, result.location};
    if (allLocationsAreHost(locations)) {
        hadamardMatricesOnHost(m1, m2, result);
    } else if (allLocationsAreDevice(locations)) {
        hadamardMatricesOnDevice(m1, m2, result);
    } else {
        throw DifferentDataLocationException();
    }
}

void transpose(const Matrix& m, Matrix& result) {
    if (m.n != result.m || m.m != result.n) {
        throw SizeMismatchException();
    }

    const std::initializer_list<DataLocation> locations = {m.location, result.location};
    if (allLocationsAreHost(locations)) {
        transposeMatrixOnHost(m, result);
    } else if (allLocationsAreDevice(locations)) {
        transposeMatrixOnDevice(m, result);
    } else {
        throw DifferentDataLocationException();
    }
}
