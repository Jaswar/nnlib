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

/**
 * @brief Allocate 1D array.
 *
 * @param n The size of the array to allocate.
 * @return Allocated array of size @p n.
 */
float* allocate1DArray(size_t n);

/**
 * @brief Allocate 1D array with a default value.
 *
 * @param n The size of the array to allocate.
 * @param defaultValue The default value to fill the array with.
 * @return Allocated array of size @p n.
 */
float* allocate1DArray(size_t n, float defaultValue);

/**
 * @brief Allocate 2D array.
 *
 * The storage of the array is row-wise.
 *
 * @param n The number of rows of the array.
 * @param m The number of columns of the array.
 * @return Allocated array of size @p n x @p m.
 */
float** allocate2DArray(size_t n, size_t m);

/**
 * @brief Allocate 2D array with a default value.
 *
 * The storage of the array is row-wise.
 *
 * @param n The number of rows of the array.
 * @param m The number of columns of the array.
 * @param defaultValue The default value to fill the array with.
 * @return Allocated array of size @p n x @p m.
 */
float** allocate2DArray(size_t n, size_t m, float defaultValue);

/**
 * @brief Copy a 1D array between two locations on host.
 *
 * @param oldLoc The data to copy.
 * @param newLoc Where to copy the data to.
 * @param n The number of elements of the data to copy.
 */
void copy1DFromHostToHost(float* oldLoc, float* newLoc, size_t n);

/**
 * @brief Copy 1D array to a new location.
 *
 * @param n The size of the array to copy.
 * @param original The array to be copied.
 * @return Copied array.
 */
float* copy1DArray(size_t n, float* original);

/**
 * @brief Copy 2D array to a new location.
 *
 * @param n The number of rows of the original array.
 * @param m The number of columns of the original array.
 * @param original The array to be copied.
 * @return Copied array.
 */
float** copy2DArray(size_t n, size_t m, float** original);

#endif //NNLIB_ALLOCATION_H
