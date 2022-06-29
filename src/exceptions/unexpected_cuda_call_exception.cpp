//
// Created by Jan Warchocki on 28/05/2022.
//

#include "unexpected_cuda_call_exception.h"

const char* UnexpectedCUDACallException::what() const noexcept {
    return "Called a CUDA only method in a non-CUDA setup";
}
