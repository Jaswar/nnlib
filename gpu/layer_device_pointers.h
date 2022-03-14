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

    DTYPE* previousWeights{};

    LayerDevicePointers(int inSize, int outSize);

    ~LayerDevicePointers();

    void allocatePreviousWeights(int n, int m);
};


#endif //NNLIB_LAYER_DEVICE_POINTERS_H
