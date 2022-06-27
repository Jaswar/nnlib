//
// Created by Jan Warchocki on 27/06/2022.
//


#include <gtest/gtest.h>
#include "utils.h"

Vector constructVector(std::initializer_list<DTYPE> vectorDefinition) {
    Vector vector = Vector(vectorDefinition.size(), HOST);

    int i = 0;
    for (auto value : vectorDefinition) {
        vector[i++] = value;
    }

    return vector;
}

Matrix constructMatrix(std::initializer_list<std::initializer_list<DTYPE>> matrixDefinition) {
    size_t numColumns = matrixDefinition.begin()->size();
    Matrix matrix = Matrix(matrixDefinition.size(), numColumns);

    int i = 0; int j = 0;
    for (auto& row : matrixDefinition) {
        for (auto value : row) {
            matrix(i, j++) = value;
        }
        i++;
        j = 0;
    }

    return matrix;
}

void assertEqual(const Matrix& result, std::initializer_list<std::initializer_list<DTYPE>> expected) {
    ASSERT_EQ(result.n, expected.size());
    ASSERT_EQ(result.m, expected.begin()->size());

    int i = 0; int j = 0;
    for (auto& row : expected) {
        for (auto value : row) {
            ASSERT_EQ(result(i, j++), value);
        }
        i++;
        j = 0;
    }
}

void assertEqual(const Vector& result, std::initializer_list<DTYPE> expected) {
    ASSERT_EQ(result.n, expected.size());

    int i = 0;
    for (auto value : expected) {
        ASSERT_EQ(result[i++], value);
    }
}



