//
// Created by Jan Warchocki on 15/03/2022.
//

#include "matrix_operations.cuh"
#include "../gpu/allocation_gpu.cuh"
#include "../gpu/verify.cuh"
#include "../gpu/assert.cuh"

__global__
void addMatricesDevice(const DTYPE* m1, const DTYPE* m2, DTYPE* result, size_t n, size_t m) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n * m) {
        return;
    }

    result[index] = m1[index] + m2[index];
}

void addMatrices(const Matrix& m1, const Matrix& m2, Matrix& result) {
    addMatricesDevice<<<m1.n, m1.m>>>(m1.data, m2.data, result.data, m1.n, m1.m);
    gpuCheckError( cudaGetLastError() )
    gpuCheckError( cudaDeviceSynchronize() )
}

__global__
void addBroadcastDevice(const DTYPE* matrix, const DTYPE* vector, DTYPE* result, size_t n, size_t m) {
    auto row = blockIdx.x;
    auto column = threadIdx.x;

    if (row > n || column > m) {
        return;
    }

    result[row * m + column] = matrix[row * m + column] + vector[column];
}

void addBroadcast(const Matrix& m, const Vector& v, Matrix& result) {
    addBroadcastDevice<<<m.n, m.m>>>(m.data, v.data, result.data, m.n, m.m);
    gpuCheckError( cudaGetLastError() )
    gpuCheckError( cudaDeviceSynchronize() )
}

__global__
void subtractMatricesDevice(const DTYPE* m1, const DTYPE* m2, DTYPE* result, size_t n, size_t m) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n * m) {
        return;
    }

    result[index] = m1[index] - m2[index];
}

void subtractMatrices(const Matrix& m1, const Matrix& m2, Matrix& result) {
    subtractMatricesDevice<<<m1.n, m1.m>>>(m1.data, m2.data, result.data, m1.n, m1.m);
    gpuCheckError( cudaGetLastError() )
    gpuCheckError( cudaDeviceSynchronize() )
}

__global__
void mulMatrixVectorDevice(const DTYPE* matrix, const DTYPE* vector, DTYPE* result, size_t n, size_t m) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    DTYPE sum = 0;
    for (int i = 0; i < m; i++) {
        sum += matrix[index * m + i] * vector[i];
    }
    result[index] = sum;
}

void multiplyMatrixVector(const Matrix& matrix, const Vector& vector, Vector& result) {
    mulMatrixVectorDevice<<<1, matrix.n>>>(matrix.data, vector.data, result.data, matrix.n, matrix.m);
    gpuCheckError( cudaGetLastError() )
    gpuCheckError( cudaDeviceSynchronize() )
}

__global__
void multiplyMatricesDevice(const DTYPE* m1, const DTYPE* m2, DTYPE* result, size_t nm1, size_t mm1, size_t mm2) {
    auto row = blockIdx.x;
    auto column = threadIdx.x;

    if (row >= nm1 || column >= mm2) {
        return;
    }

    DTYPE sum = 0;

    for (int i = 0; i < mm1; i++) {
        sum += m1[row * mm1 + i] * m2[i * mm2 + column];
    }

    result[row * mm2 + column] = sum;
}

void multiplyMatrices(const Matrix& m1, const Matrix& m2, Matrix& result) {
    multiplyMatricesDevice<<<m1.n, m2.m>>>(m1.data, m2.data, result.data, m1.n, m1.m, m2.m);
    gpuCheckError( cudaGetLastError() )
    gpuCheckError( cudaDeviceSynchronize() )
}

__global__
void multiplyMatrixDevice(const DTYPE* matrix, DTYPE constant, DTYPE* result, size_t n, size_t m) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n * m) {
        return;
    }

    result[index] = matrix[index] * constant;
}

void multiplyMatrix(const Matrix& m1, DTYPE constant, Matrix& result) {
    multiplyMatrixDevice<<<m1.n, m1.m>>>(m1.data, constant, result.data, m1.n, m1.m);
    gpuCheckError( cudaGetLastError() )
    gpuCheckError( cudaDeviceSynchronize() )
}

__global__
void transposeMatrixDevice(const DTYPE* matrix, DTYPE* result, size_t n, size_t m) {
    auto row = blockIdx.x;
    auto column = threadIdx.x;

    if (row >= n || column >= m) {
        return;
    }

    result[column * n + row] = matrix[row * m + column];
}

void transposeMatrix(const Matrix& m, Matrix& result) {
    transposeMatrixDevice<<<m.n, m.m>>>(m.data, result.data, m.n, m.m);
    gpuCheckError( cudaGetLastError() )
    gpuCheckError( cudaDeviceSynchronize() )
}
