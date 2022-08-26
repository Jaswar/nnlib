/**
 * @file tensor.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 26 August 2022
 */

#include "tensor.h"
#include "../gpu/allocation_gpu.cuh"

void Tensor::computeSize() {
    size = 1;
    for (auto it = shape.begin(); it < shape.end(); it++) {
        size *= *it;
    }
}

void Tensor::moveToDevice() {
    if (location == DEVICE) {
        return;
    }

    float* deviceData = allocate1DArray(size);
    copy1DFromHostToDevice(host, deviceData, size);

    free(host);
    device = deviceData;
    location = DEVICE;
}

void Tensor::moveToHost() {
    if (location == HOST) {
        return;
    }

    float* hostData = allocate1DArray(size);
    copy1DFromDeviceToHost(device, hostData, size);

    free(device);
    host = hostData;
    location = HOST;
}

