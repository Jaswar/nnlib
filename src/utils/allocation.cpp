/**
 * @file allocation.cpp
 * @brief Source file defining methods regarding memory allocation on host.
 * @author Jan Warchocki
 * @date 03 March 2022
 */

#include "../../include/allocation.h"
#include <cstdlib>
#include <cstring>

DTYPE** allocate2DArray(size_t n, size_t m) {
    auto data = static_cast<DTYPE**>(malloc(sizeof(DTYPE*) * n));

    for (int i = 0; i < n; i++) {
        data[i] = static_cast<DTYPE*>(malloc(sizeof(DTYPE) * m));
    }

    return data;
}

DTYPE** allocate2DArray(size_t n, size_t m, DTYPE defaultValue) {
    DTYPE** allocated = allocate2DArray(n, m);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            allocated[i][j] = defaultValue;
        }
    }

    return allocated;
}

DTYPE* allocate1DArray(size_t n) {
    return static_cast<DTYPE*>(malloc(sizeof(DTYPE) * n));
}

DTYPE* allocate1DArray(size_t n, DTYPE defaultValue) {
    DTYPE* allocated = allocate1DArray(n);

    for (int i = 0; i < n; i++) {
        allocated[i] = defaultValue;
    }

    return allocated;
}

DTYPE* copy1DArray(size_t n, DTYPE* original) {
    DTYPE* copy = allocate1DArray(n);

    memcpy(copy, original, n * sizeof(DTYPE));

    return copy;
}

DTYPE** copy2DArray(size_t n, size_t m, DTYPE** original) {
    DTYPE** copy = allocate2DArray(n, m);

    // Only the second pointer contains data, so copy only it. The first is an array of pointers.
    for (int i = 0; i < n; i++) {
        memcpy(copy[i], original[i], m * sizeof(DTYPE));
    }

    return copy;
}

void copy1DFromHostToHost(float* oldLoc, float* newLoc, size_t n) {
    memcpy(newLoc, oldLoc, n);
}
