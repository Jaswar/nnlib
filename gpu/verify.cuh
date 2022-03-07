//
// Created by Jan Warchocki on 06/03/2022.
//

#ifndef NNLIB_VERIFY_CUH
#define NNLIB_VERIFY_CUH

#if __has_include("cuda.h")

#define HAS_CUDA

#endif

bool isCudaAvailable();
void showCudaInfo();

#endif //NNLIB_VERIFY_CUH
