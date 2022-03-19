//
// Created by Jan Warchocki on 15/03/2022.
//

#include "matrix_operations.cuh"
#include "../gpu/allocation_gpu.cuh"
#include "../gpu/verify.cuh"

__global__
void addMatricesDevice(const DTYPE* m1, const DTYPE* m2, DTYPE* result, int n, int m) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n * m) {
        return;
    }

    result[index] = m1[index] + m2[index];
}

void addMatrices(const Matrix& m1, const Matrix& m2, Matrix& result) {
    addMatricesDevice<<<m1.n, m1.m>>>(m1.data, m2.data, result.data, m1.n, m1.m);
}

__global__
void subtractMatricesDevice(const DTYPE* m1, const DTYPE* m2, DTYPE* result, int n, int m) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n * m) {
        return;
    }

    result[index] = m1[index] - m2[index];
}

void subtractMatrices(const Matrix& m1, const Matrix& m2, Matrix& result) {
    subtractMatricesDevice<<<m1.n, m1.m>>>(m1.data, m2.data, result.data, m1.n, m1.m);
}

__global__
void mulMatrixVectorDevice(const DTYPE* matrix, const DTYPE* vector, DTYPE* result, int n, int m) {
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
}

__global__
void multiplyMatrixDevice(const DTYPE* matrix, DTYPE constant, DTYPE* result, int n, int m) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n * m) {
        return;
    }

    result[index] = matrix[index] * constant;
}

void multiplyMatrix(const Matrix& m1, DTYPE constant, Matrix& result) {
    multiplyMatrixDevice<<<m1.n, m1.m>>>(m1.data, constant, result.data, m1.n, m1.m);
}
