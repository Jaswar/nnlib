/**
 * @file assert.cuh
 * @brief Header file declaring the gpuAssert() method.
 * @author Jan Warchocki
 * @date 10 March 2022
 */

#ifndef NNLIB_ASSERT_CUH
#define NNLIB_ASSERT_CUH

#include "verify.cuh"
#include <cstdio>

#ifdef __CUDA__

// Taken from https://stackoverflow.com/questions/14038589
// clang-format off
/**
 * @brief Macro used to call the gpuAssert() method with file and line information.
 */
#define GPU_CHECK_ERROR(ans) gpuAssert((ans), __FILE__, __LINE__)
// clang-format on

/**
 * @brief Method used to assert that the last CUDA call was successful.
 *
 * If the call was not successful, the method will display the error message.
 *
 * @param code The most recent code returned by CUDA.
 * @param file The file where the #GPU_CHECK_ERROR was called.
 * @param line The line where the #GPU_CHECK_ERROR was called.
 * @param abort If true, the program will stop upon encountering the exception.
 */
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU assertion failed: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

#endif

#endif //NNLIB_ASSERT_CUH
