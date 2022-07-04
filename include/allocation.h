//
// Created by Jan Warchocki on 03/03/2022.
//

#ifndef NNLIB_ALLOCATION_H
#define NNLIB_ALLOCATION_H

#include <cstddef>

#define DTYPE float

DTYPE* allocate1DArray(size_t n);

DTYPE* allocate1DArray(size_t n, DTYPE DefaultValue);

DTYPE** allocate2DArray(size_t n, size_t m);

DTYPE** allocate2DArray(size_t n, size_t m, DTYPE defaultValue);

DTYPE* copy1DArray(size_t n, DTYPE* original);

DTYPE** copy2DArray(size_t n, size_t m, DTYPE** original);

#endif //NNLIB_ALLOCATION_H
