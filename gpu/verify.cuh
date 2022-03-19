//
// Created by Jan Warchocki on 06/03/2022.
//

#ifndef NNLIB_VERIFY_CUH
#define NNLIB_VERIFY_CUH

#if __has_include("cuda.h")

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define HAS_CUDA

#endif

bool isCudaAvailable();
void showCudaInfo();

#endif //NNLIB_VERIFY_CUH
