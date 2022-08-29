/**
 * @file unsupported_operation_exception.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 29 August 2022
 */

#include "unsupported_operation_exception.h"

const char* UnsupportedOperationException::what() const noexcept {
    return "This operation is not supported";
}
