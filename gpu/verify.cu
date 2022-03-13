//
// Created by Jan Warchocki on 06/03/2022.
//

#include "verify.cuh"
#include <iostream>

#ifdef HAS_CUDA

#include "cuda.h"

bool isCudaAvailable() {
    return true;
}

void showCudaInfo() {
    // Borrowed from https://stackoverflow.com/questions/5689028
    int kb = 1024;
    int mb = kb * kb;

    std::cout << "CUDA VERSION v" << CUDA_VERSION << std::endl;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Detected " << deviceCount << (deviceCount == 1 ? " device:" : " devices:") << std::endl;

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        std::cout << "DEVICE " << i << ": " << props.name << ": " << std::endl;
        std::cout << "  Compute capability:\t" << props.major << "." << props.minor << std::endl;
        std::cout << "  Global memory:\t" << props.totalGlobalMem / mb << "MB" << std::endl;
        std::cout << "  Shared memory:\t" << props.sharedMemPerBlock / kb << "KB" << std::endl;
        std::cout << "  Constant memory:\t" << props.totalConstMem / kb << "KB" << std::endl;
        std::cout << "  Block registers:\t" << props.regsPerBlock << std::endl;

        std::cout << "  Warp size:\t\t" << props.warpSize << std::endl;
        std::cout << "  Threads per block:\t" << props.maxThreadsPerBlock << std::endl;
        std::cout << "  Max block dimensions:\t[ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << std::endl;
        std::cout << "  Max grid dimensions:\t[ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << std::endl;
    }

    std::cout << std::endl;
}

#else

bool isCudaAvailable() {
    return false;
}

void showCudaInfo() {
    std::cout << "No version of CUDA is available." << std::endl << std::endl;
}

#endif