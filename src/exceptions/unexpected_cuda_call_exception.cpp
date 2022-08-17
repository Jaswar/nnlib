/**
 * @file unexpected_cuda_call_exception.cpp
 * @brief Source file defining the UnexpectedCUDACallException class.
 * @author Jan Warchocki
 * @date 28 May 2022
 */

#include "unexpected_cuda_call_exception.h"

const char* UnexpectedCUDACallException::what() const noexcept {
    return "Called a CUDA only method in a non-CUDA setup";
}
