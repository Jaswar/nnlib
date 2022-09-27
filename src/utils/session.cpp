/**
 * @file session.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 29 August 2022
 */

#include "session.h"
#include "verify.cuh"

Session::Session() {
    numCores = std::thread::hardware_concurrency();
#ifdef HAS_CUDA
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    threadsPerBlock = props.maxThreadsPerBlock;
#else
    threadsPerBlock = 0;
#endif
}