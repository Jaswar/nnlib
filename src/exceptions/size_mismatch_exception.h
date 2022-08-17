/**
 * @file size_mismatch_exception.h
 * @brief Header file declaring the SizeMismatchException class.
 * @author Jan Warchocki
 * @date 03 March 2022
 */

#ifndef NNLIB_SIZE_MISMATCH_EXCEPTION_H
#define NNLIB_SIZE_MISMATCH_EXCEPTION_H

#include <exception>

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
    const char* what() const noexcept override;
};


#endif //NNLIB_SIZE_MISMATCH_EXCEPTION_H
