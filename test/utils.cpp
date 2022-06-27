//
// Created by Jan Warchocki on 27/06/2022.
//


#include "utils.h"

Vector constructVector(std::initializer_list<DTYPE> vectorDefinition) {
    Vector vector = Vector(vectorDefinition.size(), HOST);

    size_t i = 0;
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



