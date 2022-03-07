//
// Created by Jan Warchocki on 03/03/2022.
//

#include "vector.h"
#include "../exceptions/size_mismatch_exception.h"

Vector::Vector(int n) : Vector(allocate1DArray(n), n) {}

Vector::Vector(DTYPE* data, int n) : data(data), n(n) {}

Vector::Vector(const Vector& vector) {
    n = vector.n;
    data = copy1DArray(n, vector.data);
}

Vector::~Vector() {
    free(data);
}

Vector& Vector::operator=(const Vector& vector) {
    free(data);

    n = vector.n;
    data = copy1DArray(n, vector.data);

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

    DTYPE* newData = allocate1DArray(v1.n);

    for (int i = 0; i < v1.n; i++) {
        newData[i] = v1[i] + v2[i];
    }

    return Vector(newData, v1.n);
}

Vector operator-(const Vector& v1, const Vector& v2) {
    if (v1.n != v2.n) {
        throw SizeMismatchException();
    }

    DTYPE* newData = allocate1DArray(v1.n);

    for (int i = 0; i < v1.n; i++) {
        newData[i] = v1[i] - v2[i];
    }

    return Vector(newData, v1.n);
}

Vector operator*(const Vector& v1, DTYPE constant) {
    DTYPE* newData = allocate1DArray(v1.n);

    for (int i = 0; i < v1.n; i++) {
        newData[i] = v1[i] * constant;
    }

    return Vector(newData, v1.n);
}

Vector operator*(DTYPE constant, const Vector& v1) {
    return v1 * constant;
}

DTYPE operator*(const Vector& v1, const Vector& v2) {
    if (v1.n != v2.n) {
        throw SizeMismatchException();
    }

    DTYPE dotProduct = 0;
    for (int i = 0; i < v1.n; i++) {
        dotProduct += v1[i] * v2[i];
    }

    return dotProduct;
}

