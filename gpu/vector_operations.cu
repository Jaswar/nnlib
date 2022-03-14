//
// Created by Jan Warchocki on 14/03/2022.
//

#include "vector_operations.cuh"
#include "allocation_gpu.cuh"

__global__
void addVectorsDevice(const DTYPE* v1, const DTYPE* v2, DTYPE* result, int n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    result[index] = v1[index] + v2[index];
}

Vector addVectors(const Vector& v1, const Vector& v2) {
    DTYPE* result = allocate1DArrayDevice(v1.n);

    addVectorsDevice<<<1, v1.n>>>(v1.data, v2.data, result, v1.n);

    return Vector(result, v1.n, DEVICE);
}

__global__
void subtractVectorsDevice(const DTYPE* v1, const DTYPE* v2, DTYPE* result, int n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    result[index] = v1[index] - v2[index];
}


Vector subtractVectors(const Vector& v1, const Vector& v2) {
    DTYPE* result = allocate1DArrayDevice(v1.n);

    subtractVectorsDevice<<<1, v1.n>>>(v1.data, v2.data, result, v1.n);

    return Vector(result, v1.n, DEVICE);
}

__global__
void multiplyVectorDevice(const DTYPE* v1, DTYPE constant, DTYPE* result, int n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    result[index] = v1[index] * constant;
}

Vector multiplyVector(const Vector& v1, float constant) {
    DTYPE* result = allocate1DArrayDevice(v1.n);

    multiplyVectorDevice<<<1, v1.n>>>(v1.data, constant, result, v1.n);

    return Vector(result, v1.n, DEVICE);
}


