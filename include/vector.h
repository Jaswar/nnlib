//
// Created by Jan Warchocki on 03/03/2022.
//

#ifndef NNLIB_VECTOR_H
#define NNLIB_VECTOR_H

#include <iostream>
#include "allocation.h"

enum DataLocation {HOST, DEVICE};

class Vector {

public:
    DTYPE* data;
    size_t n;
    DataLocation location;

    explicit Vector(size_t n);
    Vector(size_t n, DataLocation location);
    Vector(DTYPE* data, size_t n);
    Vector(DTYPE* data, size_t n, DataLocation location);

    Vector(const Vector& vector);

    ~Vector();

    void moveToDevice();
    void moveToHost();

    Vector& operator=(const Vector& other);
    DTYPE& operator[](size_t index) const;
};

// "toString" for std::cout
std::ostream& operator<<(std::ostream& stream, const Vector& vector);

void add(const Vector& v1, const Vector& v2, Vector& result);
void subtract(const Vector& v1, const Vector& v2, Vector& result);
void multiply(const Vector& v1, DTYPE constant, Vector& result);
void multiply(DTYPE constant, const Vector& v2, Vector& result);

#endif //NNLIB_VECTOR_H
