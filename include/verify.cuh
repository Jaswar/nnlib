/**
 * @file verify.cuh
 * @brief Header file declaring methods used for verifying the existence of a CUDA runtime.
 * @author Jan Warchocki
 * @date 06 March 2022
 */

#ifndef NNLIB_VERIFY_CUH
#define NNLIB_VERIFY_CUH

#if __has_include("cuda.h") || defined RUN_DOCUMENTATION

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * @brief Macro defined if CUDA is available.
 *
 * Used when defining methods to define CUDA code if and only if CUDA is available.
 */
#define HAS_CUDA

#endif

/**
 * @brief Check if CUDA is available.
 *
 * The method is implemented twice depending on if #HAS_CUDA is defined. If it is, then the method returns true,
 * otherwise it returns false.
 *
 * @return True if CUDA is available, false otherwise.
 */
bool isCudaAvailable();

/**
 * @brief Show information about CUDA and the available GPU(s).
 */
void showCudaInfo();

#endif //NNLIB_VERIFY_CUH
