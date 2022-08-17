/**
 * @file different_data_location_exception.cpp
 * @brief Source file defining the DifferentDataLocationException class.
 * @author Jan Warchocki
 * @date 14 March 2022
 */

#include "different_data_location_exception.h"

const char* DifferentDataLocationException::what() const noexcept {
    return "Different data location exception";
}
