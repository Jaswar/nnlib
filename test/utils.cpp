//
// Created by Jan Warchocki on 27/06/2022.
//

#include "utils.h"

Vector constructVector(std::initializer_list<DTYPE> vectorDefinition, DataLocation location) {
    Vector vector = Vector(vectorDefinition.size(), HOST);

    int i = 0;
    for (auto value : vectorDefinition) {
        vector[i++] = value;
    }

    if (location == DEVICE) {
        vector.moveToDevice();
    }

    return vector;
}

Matrix constructMatrix(std::initializer_list<std::initializer_list<DTYPE>> matrixDefinition, DataLocation location) {
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

    if (location == DEVICE) {
        matrix.moveToDevice();
    }

    return matrix;
}





