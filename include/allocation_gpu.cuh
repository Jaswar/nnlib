/**
 * @file allocation_gpu.cuh
 * @brief Header file to declare common functions regarding memory allocation on device.
 *
 * All functions in the file only allocate/copy float memory.
 *
 * @author Jan Warchocki
 * @date 10 March 2022
 */

#ifndef NNLIB_ALLOCATION_GPU_CUH
#define NNLIB_ALLOCATION_GPU_CUH

#include "allocation.h"
#include "assert.cuh"
#include "verify.cuh"
#include "exceptions.h"

#ifdef __CUDA__

/**
 * @brief Allocate a 1D array.
 *
 * @param n The size of the array to allocate.
 * @return The allocated array.
 */
template<typename T>
T* allocate1DArrayDevice(size_t n) {
    T* allocated;
    GPU_CHECK_ERROR(cudaMalloc(&allocated, n * sizeof(T)));
    return allocated;
}

/**
 * @brief Copy a 1D array within device memory.
 *
 * @param oldLoc The old location of the array.
 * @param newLoc The new location of the array.
 * @param n The size of the array.
 */
template<typename T>
void copy1DFromDeviceToDevice(T* oldLoc, T* newLoc, size_t n) {
    GPU_CHECK_ERROR(cudaMemcpy(newLoc, oldLoc, n * sizeof(T), cudaMemcpyDeviceToDevice));
}

/**
 * @brief Copy a 1D array from host memory to device memory.
 *
 * @param host The location of the host array.
 * @param device The device location where the array should be copied.
 * @param n The size of the array.
 */
template<typename T>
void copy1DFromHostToDevice(T* host, T* device, size_t n) {
    GPU_CHECK_ERROR(cudaMemcpy(device, host, n * sizeof(T), cudaMemcpyHostToDevice));
}

/**
 * @brief Copy a 1D array from device memory to host memory.
 *
 * @param device The location of the device array.
 * @param host The host location where the array should be copied.
 * @param n The size of the array.
 */
template<typename T>
void copy1DFromDeviceToHost(T* device, T* host, size_t n) {
    GPU_CHECK_ERROR(cudaMemcpy(host, device, n * sizeof(T), cudaMemcpyDeviceToHost));
}

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
template<typename T>
T* copy1DArrayDevice(T* old, size_t n) {
    T* allocated = allocate1DArrayDevice<T>(n);
    GPU_CHECK_ERROR(cudaMemcpy(allocated, old, n * sizeof(T), cudaMemcpyDeviceToDevice));
    return allocated;
}

template<typename T>
void copy1DArrayDevice(T* old, T* copy, size_t n) {
    GPU_CHECK_ERROR(cudaMemcpy(copy, old, n * sizeof(T), cudaMemcpyDeviceToDevice));
}

/**
 * @brief Free a 1D array from device memory.
 *
 * @param device The array to free.
 */
template<typename T>
void free1DArrayDevice(T* device) {
    GPU_CHECK_ERROR(cudaFree(device));
}

#else

template<typename T>
T* allocate1DArrayDevice(size_t n) {
    throw UnexpectedCUDACallException();
}

template<typename T>
void copy1DFromDeviceToDevice(T* oldLoc, T* newLoc, size_t n) {
    throw UnexpectedCUDACallException();
}

template<typename T>
void copy1DFromHostToDevice(T* host, T* device, size_t n) {
    throw UnexpectedCUDACallException();
}


template<typename T>
void copy1DFromDeviceToHost(T* device, T* host, size_t n) {
    throw UnexpectedCUDACallException();
}

template<typename T>
T* copy1DArrayDevice(size_t n, T* old) {
    throw UnexpectedCUDACallException();
}

template<typename T>
void copy1DArrayDevice(size_t n, T* old, T* copy) {
    throw UnexpectedCUDACallException();
}

template<typename T>
void free1DArrayDevice(T* device) {
    throw UnexpectedCUDACallException();
}

#endif

#endif //NNLIB_ALLOCATION_GPU_CUH
