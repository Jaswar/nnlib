//
// Created by Jan Warchocki on 14/03/2022.
//

#include "different_data_location_exception.h"

const char* DifferentDataLocationException::what() const noexcept {
    return "Different data location exception";
}
