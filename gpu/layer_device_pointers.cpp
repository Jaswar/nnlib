//
// Created by Jan Warchocki on 10/03/2022.
//

#include "layer_device_pointers.h"
#include "verify.cuh"
#include "allocation_gpu.cuh"


LayerDevicePointers::LayerDevicePointers(int inSize, int outSize) : inSize(inSize), outSize(outSize) {
    if (isCudaAvailable()) {
        weights = allocate1DArrayDevice(inSize * outSize);
    }
}

LayerDevicePointers::~LayerDevicePointers() {
    // TODO: fix this so it frees the memory
//    free1DArrayDevice(weights);
//    free1DArrayDevice(biases);
//    free1DArrayDevice(data);
//    free1DArrayDevice(derivatives);
//
//    free1DArrayDevice(newDelta);
//
//    if (delta != nullptr) {
//        free1DArrayDevice(delta);
//    }
//    if (previousWeights != nullptr) {
//        free1DArrayDevice(previousWeights);
//    }
}

void LayerDevicePointers::allocatePreviousWeights(int n, int m) {
    if (isCudaAvailable() && previousWeights == nullptr && n > 0 && m > 0) {
        previousWeights = allocate1DArrayDevice(n * m);
    }
}
