//
// Created by Jan Warchocki on 03/03/2022.
//

#ifndef NNLIB_ALLOCATION_H
#define NNLIB_ALLOCATION_H

#define DTYPE float

DTYPE* allocate1DArray(int n);

DTYPE* allocate1DArray(int n, DTYPE defaultValue);

DTYPE** allocate2DArray(int n, int m);

DTYPE** allocate2DArray(int n, int m, DTYPE defaultValue);

DTYPE* copy1DArray(int n, DTYPE* original);

DTYPE** copy2DArray(int n, int m, DTYPE** original);

#endif //NNLIB_ALLOCATION_H
