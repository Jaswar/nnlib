//
// Created by Jan Warchocki on 10/03/2022.
//

#include "allocation_gpu.cuh"
#include "verify.cuh"

#ifdef HAS_CUDA

DTYPE* allocate1DArrayDevice(int n) {
    DTYPE* allocated;
    cudaMalloc(&allocated, n * sizeof(DTYPE));
    return allocated;
}

void copy1DFromHostToDevice(DTYPE* host, DTYPE* device, int n) {
    cudaMemcpy(device, host, n * sizeof(DTYPE), cudaMemcpyHostToDevice);
}

void copy2DFromHostToDevice(DTYPE** host, DTYPE* device, int n, int m) {
    DTYPE* temp = allocate1DArray(n * m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            temp[i * m + j] = host[i][j];
        }
    }
    cudaMemcpy(device, temp, n * m * sizeof(DTYPE), cudaMemcpyHostToDevice);
    free(temp);
}

void free1DArrayDevice(DTYPE* device) {
    cudaFree(device);
}

void copy1DFromDeviceToHost(DTYPE* device, DTYPE* host, int n) {
    cudaMemcpy(host, device, n * sizeof(DTYPE), cudaMemcpyDeviceToHost);
}

void copy2DFromDeviceToHost(DTYPE* device, DTYPE** host, int n, int m) {
    DTYPE* temp = allocate1DArray(n * m);
    copy1DFromDeviceToHost(device, temp, n * m);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            DTYPE val = temp[i * m + j];
            host[i][j] = val;
        }
    }

    free(temp);
}

#else

DTYPE* allocate1DArrayDevice(int n) {
    return nullptr;
}

DTYPE** allocate2DArrayDevice(int n, int m) {
    return nullptr;
}

void copy1DFromHostToDevice(DTYPE* host, DTYPE* device, int n) {

}

void copy2DFromHostToDevice(DTYPE** host, DTYPE** device, int n, int m) {

}

#endif