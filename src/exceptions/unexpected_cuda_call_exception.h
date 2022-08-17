/**
 * @file unexpected_cuda_call_exception.h
 * @brief Header file declaring the UnexpectedCUDACallException class.
 * @author Jan Warchocki
 * @date 28 May 2022
 */

#ifndef NNLIB_UNEXPECTED_CUDA_CALL_EXCEPTION_H
#define NNLIB_UNEXPECTED_CUDA_CALL_EXCEPTION_H

#include <exception>

/**
 * @brief Exception to be thrown when a CUDA method was called despite no CUDA/GPU support.
 */
class UnexpectedCUDACallException : public std::exception {

    /**
     * @brief Return the exception description.
     *
     * @return The exception description.
     */
    const char* what() const noexcept override;
};


#endif //NNLIB_UNEXPECTED_CUDA_CALL_EXCEPTION_H
