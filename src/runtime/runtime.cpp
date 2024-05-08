/**
 * @file runtime.cpp
 * @brief
 *
 * @author Jan Warchocki
 * @date 08 May 2024
 *
 */

#include "runtime.h"

Runtime::Runtime() {
    useGradient = true;
}

void Runtime::disableGradient() {
    useGradient = false;
}

void Runtime::enableGradient() {
    useGradient = true;
}

