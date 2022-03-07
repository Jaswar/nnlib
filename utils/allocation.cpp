//
// Created by Jan Warchocki on 03/03/2022.
//

#include <cstdlib>
#include <cstring>
#include "allocation.h"

DTYPE** allocate2DArray(int n, int m) {
    auto data = (DTYPE **) malloc(sizeof(DTYPE*) * n);

    for (int i = 0; i < n; i++) {
        data[i] = (DTYPE *) malloc(sizeof(DTYPE) * m);
    }

    return data;
}

DTYPE** allocate2DArray(int n, int m, DTYPE defaultValue) {
    DTYPE** allocated = allocate2DArray(n, m);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            allocated[i][j] = defaultValue;
        }
    }

    return allocated;
}

DTYPE* allocate1DArray(int n) {
    return (DTYPE*) malloc(sizeof(DTYPE) * n);
}

DTYPE* allocate1DArray(int n, DTYPE defaultValue) {
    DTYPE* allocated = allocate1DArray(n);

    for (int i = 0; i < n; i++) {
        allocated[i] = defaultValue;
    }

    return allocated;
}

DTYPE* copy1DArray(int n, DTYPE* original) {
    DTYPE* copy = allocate1DArray(n);

    memcpy(copy, original, n * sizeof(DTYPE));

    return copy;
}

DTYPE** copy2DArray(int n, int m, DTYPE** original) {
    DTYPE** copy = allocate2DArray(n, m);

    // Only the second pointer contains data, so copy only it. The first is an array of pointers.
    for (int i = 0; i < n; i++) {
        memcpy(copy[i], original[i], m * sizeof(DTYPE));
    }

    return copy;
}

