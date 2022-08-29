/**
 * @file unsupported_operation_exception.h
 * @brief 
 * @author Jan Warchocki
 * @date 29 August 2022
 */

#ifndef NNLIB_UNSUPPORTED_OPERATION_EXCEPTION_H
#define NNLIB_UNSUPPORTED_OPERATION_EXCEPTION_H

#include <exception>

class UnsupportedOperationException : public std::exception {
    const char* what() const noexcept override;
};


#endif //NNLIB_UNSUPPORTED_OPERATION_EXCEPTION_H
