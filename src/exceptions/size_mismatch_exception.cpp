/**
 * @file size_mismatch_exception.cpp
 * @brief Source file defining the SizeMismatchException class.
 * @author Jan Warchocki
 * @date 03 March 2022
 */

#include "size_mismatch_exception.h"

const char* SizeMismatchException::what() const noexcept {
    return "Size mismatch exception";
}
