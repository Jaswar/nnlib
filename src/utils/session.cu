/**
 * @file session.cu
 * @brief Source file defining the Session class.
 * @author Jan Warchocki
 * @date 29 August 2022
 */

#include "session.cuh"
#include "verify.cuh"

Session::Session() {
    numCores = 32;
#ifdef __CUDA__
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    //    cudaDeviceProp props;
    //    cudaGetDeviceProperties(&props, 0);
    threadsPerBlock = 1024;
#else
    threadsPerBlock = 0;
#endif
}
