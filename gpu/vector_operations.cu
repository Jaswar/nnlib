//
// Created by Jan Warchocki on 14/03/2022.
//

#include "vector_operations.cuh"
#include "allocation_gpu.cuh"
#include "verify.cuh"

__global__
void addVectorsDevice(const DTYPE* v1, const DTYPE* v2, DTYPE* result, int n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    result[index] = v1[index] + v2[index];
}

void addVectors(const Vector& v1, const Vector& v2, Vector& result) {
    addVectorsDevice<<<1, v1.n>>>(v1.data, v2.data, result.data, v1.n);
}

__global__
void subtractVectorsDevice(const DTYPE* v1, const DTYPE* v2, DTYPE* result, int n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    result[index] = v1[index] - v2[index];
}


void subtractVectors(const Vector& v1, const Vector& v2, Vector& result) {
    subtractVectorsDevice<<<1, v1.n>>>(v1.data, v2.data, result.data, v1.n);
}

__global__
void multiplyVectorDevice(const DTYPE* v1, DTYPE constant, DTYPE* result, int n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    result[index] = v1[index] * constant;
}

void multiplyVector(const Vector& v1, DTYPE constant, Vector& result) {
    multiplyVectorDevice<<<1, v1.n>>>(v1.data, constant, result.data, v1.n);
}


