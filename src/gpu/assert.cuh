//
// Created by Jan Warchocki on 10/03/2022.
//

#ifndef NNLIB_ASSERT_CUH
#define NNLIB_ASSERT_CUH

#include "verify.cuh"
#include <cstdio>

#ifdef HAS_CUDA

// Taken from https://stackoverflow.com/questions/14038589
// clang-format off
#define GPU_CHECK_ERROR(ans) gpuAssert((ans), __FILE__, __LINE__)
// clang-format on
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU assertion failed: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#endif

#endif //NNLIB_ASSERT_CUH
