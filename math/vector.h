//
// Created by Jan Warchocki on 03/03/2022.
//

#ifndef NNLIB_VECTOR_H
#define NNLIB_VECTOR_H

#include <iostream>
#include "../utils/allocation.h"

enum dLocation {HOST, DEVICE};

class Vector {

public:
    DTYPE* data;
    int n;
    dLocation location;

    explicit Vector(int n);
    Vector(int n, dLocation location);
    Vector(DTYPE* data, int n);
    Vector(DTYPE* data, int n, dLocation location);

    Vector(const Vector& vector);

    ~Vector();

    void moveToDevice();
    void moveToHost();

    Vector& operator=(const Vector& other);
    DTYPE operator[](int index) const;
    DTYPE& operator[](int index);
};

// "toString" for std::cout
std::ostream& operator<<(std::ostream& stream, const Vector& vector);

Vector operator+(const Vector& v1, const Vector& v2);
Vector operator-(const Vector& v1, const Vector& v2);
Vector operator*(const Vector& v1, DTYPE constant);
Vector operator*(DTYPE constant, const Vector& v1);

#endif //NNLIB_VECTOR_H
