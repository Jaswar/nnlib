//
// Created by Jan Warchocki on 03/03/2022.
//

#ifndef NNLIB_VECTOR_H
#define NNLIB_VECTOR_H

#include <iostream>
#include "../utils/allocation.h"

class Vector {

public:
    DTYPE* data;
    int n;

    explicit Vector(int n);
    Vector(DTYPE* data, int n);
    Vector(const Vector& vector);

    ~Vector();

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
// Dot product
DTYPE operator*(const Vector& v1, const Vector& v2);

#endif //NNLIB_VECTOR_H
