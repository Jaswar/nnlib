/**
 * @file allocation.h
 * @brief Header file to declare common functions regarding memory allocation.
 *
 * All functions in the file only allocate/copy memory of type #DTYPE.
 *
 * @author Jan Warchocki
 * @date 03/03/2022
 *
 */

#ifndef NNLIB_ALLOCATION_H
#define NNLIB_ALLOCATION_H

#include <cstddef>

/**
 * @brief Data type used throughout the library as the main storage and computation type.
 */
#define DTYPE float

/**
 * @brief Allocate 1D array.
 *
 * @param n The size of the array to allocate.
 * @return Allocated #DTYPE array of size @p n.
 */
DTYPE* allocate1DArray(size_t n);

/**
 * @brief Allocate 1D array with a default value.
 *
 * @param n The size of the array to allocate.
 * @param defaultValue The default value to fill the array with.
 * @return Allocated #DTYPE array of size @p n.
 */
DTYPE* allocate1DArray(size_t n, DTYPE defaultValue);

/**
 * @brief Allocate 2D array.
 *
 * The storage of the array is row-wise.
 *
 * @param n The number of rows of the array.
 * @param m The number of columns of the array.
 * @return Allocated #DTYPE array of size @p n x @p m.
 */
DTYPE** allocate2DArray(size_t n, size_t m);

/**
 * @brief Allocate 2D array with a default value.
 *
 * The storage of the array is row-wise.
 *
 * @param n The number of rows of the array.
 * @param m The number of columns of the array.
 * @param defaultValue The default value to fill the array with.
 * @return Allocated #DTYPE array of size @p n x @p m.
 */
DTYPE** allocate2DArray(size_t n, size_t m, DTYPE defaultValue);

/**
 * @brief Copy 1D array to a new location.
 *
 * @param n The size of the array to copy.
 * @param original The array to be copied.
 * @return Copied array.
 */
DTYPE* copy1DArray(size_t n, DTYPE* original);

/**
 * @brief Copy 2D array to a new location.
 *
 * @param n The number of rows of the original array.
 * @param m The number of columns of the original array.
 * @param original The array to be copied.
 * @return Copied array.
 */
DTYPE** copy2DArray(size_t n, size_t m, DTYPE** original);

#endif //NNLIB_ALLOCATION_H
