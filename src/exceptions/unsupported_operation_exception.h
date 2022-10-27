/**
 * @file unsupported_operation_exception.h
 * @brief Header file declaring the UnsupportedOperationException class.
 * @author Jan Warchocki
 * @date 29 August 2022
 */

#ifndef NNLIB_UNSUPPORTED_OPERATION_EXCEPTION_H
#define NNLIB_UNSUPPORTED_OPERATION_EXCEPTION_H

#include <exception>

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
    const char* what() const noexcept override;
};


#endif //NNLIB_UNSUPPORTED_OPERATION_EXCEPTION_H
