/**
 * @file allocation.h
 * @brief Header file to declare common functions regarding memory allocation on host.
 *
 * All functions in the file only allocate/copy float memory.
 *
 * @author Jan Warchocki
 * @date 03 March 2022
 *
 */

#ifndef NNLIB_ALLOCATION_H
#define NNLIB_ALLOCATION_H

#include <cstddef>
#include <cstdlib>
#include <cstring>

/**
 * @brief Allocate 1D array.
 *
 * @param n The size of the array to allocate.
 * @return Allocated array of size @p n.
 */
template <typename T>
T* allocate1DArray(size_t n) {
    return static_cast<T*>(malloc(sizeof(float) * n));
}

/**
 * @brief Copy a 1D array between two locations on host.
 *
 * @param oldLoc The data to copy.
 * @param newLoc Where to copy the data to.
 * @param n The number of elements of the data to copy.
 */
template <typename T>
void copy1DArrayHost(T* oldLoc, T* newLoc, size_t n) {
    memcpy(newLoc, oldLoc, n * sizeof(T));
}

/**
 * @brief Copy 1D array to a new location.
 *
 * @param n The size of the array to copy.
 * @param original The array to be copied.
 * @return Copied array.
 */
template <typename T>
T* copy1DArrayHost(T* original, size_t n) {
    T* copy = allocate1DArray<T>(n);
    memcpy(copy, original, n * sizeof(T));
    return copy;
}

#endif //NNLIB_ALLOCATION_H
