//
// Created by Jan Warchocki on 27/06/2022.
//

#include "assertions.h"

#include <utility>

bool withinBoundsAbsolute(const float v1, const float v2, const float delta) {
    return std::abs(v1 - v2) <= delta;
}

bool withinBoundsRelative(const float v1, const float v2, const float delta) {
    if (v2 == 0)
        return v1 == 0;
    return std::abs(v1 / v2 - 1) <= delta;
}


bool withinBounds(float v1, float v2, float delta, bool relative) {
    if (relative) {
        return withinBoundsRelative(v1, v2, delta);
    } else {
        return withinBoundsAbsolute(v1, v2, delta);
    }
}