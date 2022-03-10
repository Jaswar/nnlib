//
// Created by Jan Warchocki on 10/03/2022.
//

#ifndef NNLIB_LAYER_DEVICE_POINTERS_H
#define NNLIB_LAYER_DEVICE_POINTERS_H


#include "../utils/allocation.h"

class LayerDevicePointers {
    int inSize;
    int outSize;

public:
    DTYPE* weights;
    DTYPE* biases;
    DTYPE* data;

    DTYPE* derivatives;
    DTYPE* delta{};

    DTYPE* previousWeights{};

    DTYPE* newDelta;

    LayerDevicePointers(int inSize, int outSize);

    ~LayerDevicePointers();

    void allocatePreviousWeights(int n, int m);

    void allocateDelta(int n);
};


#endif //NNLIB_LAYER_DEVICE_POINTERS_H
