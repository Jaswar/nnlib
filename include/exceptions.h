/**
 * @file exceptions.h
 * @brief
 *
 * @author Jan Warchocki
 * @date 20 May 2024
 *
 */

#ifndef NNLIB_EXCEPTIONS_H
#define NNLIB_EXCEPTIONS_H

#include <exception>

/**
 * @brief Exception to be thrown where operands are located in different places.
 *
 * For example, when performing the add operation on two matrices, and one of them is located on host
 * and one on device, this exception should be thrown.
 */
class DifferentDataLocationException : public std::exception {

    /**
     * @brief Return the exception description.
     *
     * @return The exception description.
     */
    [[nodiscard]] const char* what() const noexcept override;
};

/**
 * @brief Exception to be thrown where operands are different shapes.
 *
 * For example, when performing the add operation on two matrices, which are different shapes,
 * this exception should be thrown.
 */
class SizeMismatchException : public std::exception {

    /**
     * @brief Return the exception description.
     *
     * @return The exception description.
     */
    [[nodiscard]] const char* what() const noexcept override;
};

/**
 * @brief Exception to be thrown when a CUDA method was called despite no CUDA/GPU support.
 */
class UnexpectedCUDACallException : public std::exception {

    /**
     * @brief Return the exception description.
     *
     * @return The exception description.
     */
    [[nodiscard]] const char* what() const noexcept override;
};

/**
 * @brief Exception to be thrown when an invalid operation is to be performed.
 *
 * This example will be thrown when, for example, a 3D tensor will be attempted to be multiplied with another 3D
 * tensor. Such an operation is not yet defined in the library and hence is invalid.
 */
class UnsupportedOperationException : public std::exception {

    /**
     * @brief Return the exception description.
     *
     * @return The exception description.
     */
    [[nodiscard]] const char* what() const noexcept override;
};


#endif //NNLIB_EXCEPTIONS_H
