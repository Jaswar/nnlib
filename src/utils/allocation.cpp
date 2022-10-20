/**
 * @file allocation.cpp
 * @brief Source file defining methods regarding memory allocation on host.
 * @author Jan Warchocki
 * @date 03 March 2022
 */

#include "../../include/allocation.h"
#include <cstdlib>
#include <cstring>

float** allocate2DArray(size_t n, size_t m) {
    auto data = static_cast<float**>(malloc(sizeof(float*) * n));

    for (int i = 0; i < n; i++) {
        data[i] = static_cast<float*>(malloc(sizeof(float) * m));
    }

    return data;
}

float** allocate2DArray(size_t n, size_t m, float defaultValue) {
    float** allocated = allocate2DArray(n, m);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            allocated[i][j] = defaultValue;
        }
    }

    return allocated;
}

float* allocate1DArray(size_t n) {
    return static_cast<float*>(malloc(sizeof(float) * n));
}

float* allocate1DArray(size_t n, float defaultValue) {
    float* allocated = allocate1DArray(n);

    for (int i = 0; i < n; i++) {
        allocated[i] = defaultValue;
    }

    return allocated;
}

float* copy1DArray(size_t n, float* original) {
    float* copy = allocate1DArray(n);

    memcpy(copy, original, n * sizeof(float));

    return copy;
}

float** copy2DArray(size_t n, size_t m, float** original) {
    float** copy = allocate2DArray(n, m);

    // Only the second pointer contains data, so copy only it. The first is an array of pointers.
    for (int i = 0; i < n; i++) {
        memcpy(copy[i], original[i], m * sizeof(float));
    }

    return copy;
}

void copy1DFromHostToHost(float* oldLoc, float* newLoc, size_t n) {
    memcpy(newLoc, oldLoc, n * sizeof(float));
}
