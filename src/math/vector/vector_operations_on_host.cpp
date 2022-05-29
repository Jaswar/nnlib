//
// Created by Jan Warchocki on 29/05/2022.
//

#include "vector_operations_on_host.h"

void addVectorsOnHost(const Vector& v1, const Vector& v2, Vector& result) {
    for (int i = 0; i < v1.n; i++) {
        result[i] = v1[i] + v2[i];
    }
}

void subtractVectorsOnHost(const Vector& v1, const Vector& v2, Vector& result) {
    for (int i = 0; i < v1.n; i++) {
        result[i] = v1[i] - v2[i];
    }
}

void multiplyVectorOnHost(const Vector& v1, DTYPE constant, Vector& result) {
    for (int i = 0; i < v1.n; i++) {
        result[i] = v1[i] * constant;
    }
}
