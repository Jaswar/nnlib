/**
 * @file allocation_gpu.cuh
 * @brief Header file to declare common functions regarding memory allocation on device.
 *
 * All functions in the file only allocate/copy memory of type #DTYPE.
 *
 * @author Jan Warchocki
 * @date 10 March 2022
 */

#ifndef NNLIB_ALLOCATION_GPU_CUH
#define NNLIB_ALLOCATION_GPU_CUH

#include "../../include/allocation.h"

/**
 * @brief Allocate a 1D array.
 *
 * @param n The size of the array to allocate.
 * @return The allocated array.
 */
DTYPE* allocate1DArrayDevice(size_t n);

/**
 * @brief Copy a 1D array within device memory.
 *
 * @param oldLoc The old location of the array.
 * @param newLoc The new location of the array.
 * @param n The size of the array.
 */
void copy1DFromDeviceToDevice(DTYPE* oldLoc, DTYPE* newLoc, size_t n);

/**
 * @brief Copy a 1D array from host memory to device memory.
 *
 * @param host The location of the host array.
 * @param device The device location where the array should be copied.
 * @param n The size of the array.
 */
void copy1DFromHostToDevice(DTYPE* host, DTYPE* device, size_t n);

/**
 * @brief Copy a 2D array from host memory to device memory.
 *
 * @param host The location of the host array.
 * @param device The device location where the array should be copied.
 * @param n The number of rows of the array.
 * @param m The number of columns of the array.
 */
void copy2DFromHostToDevice(DTYPE** host, DTYPE* device, size_t n, size_t m);

/**
 * @brief Copy a 1D array from device memory to host memory.
 *
 * @param device The location of the device array.
 * @param host The host location where the array should be copied.
 * @param n The size of the array.
 */
void copy1DFromDeviceToHost(DTYPE* device, DTYPE* host, size_t n);

/**
 * @brief Copy a 2D array from device memory to host memory.
 *
 * @param device The location of the device array.
 * @param host The host location where the array should be copied.
 * @param n The number of rows of the array.
 * @param m The number of columns of the array.
 */
void copy2DFromDeviceToHost(DTYPE* device, DTYPE** host, size_t n, size_t m);

/**
 * @brief Copy the provided 1D array to a new location.
 *
 * As opposed to copy1DFromDeviceToDevice(), this returns the new location, rather than
 * expecting it as an argument.
 *
 * @param n The size of the array to copy.
 * @param old The array to copy.
 * @return The new array.
 */
DTYPE* copy1DArrayDevice(size_t n, DTYPE* old);

/**
 * @brief Free a 1D array from device memory.
 *
 * @param device The array to free.
 */
void free1DArrayDevice(DTYPE* device);

#endif //NNLIB_ALLOCATION_GPU_CUH
