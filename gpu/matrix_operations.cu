//
// Created by Jan Warchocki on 15/03/2022.
//

#include "matrix_operations.cuh"
#include "allocation_gpu.cuh"

__global__
void addMatricesDevice(const DTYPE* m1, const DTYPE* m2, DTYPE* result, int n, int m) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n * m) {
        return;
    }

    result[index] = m1[index] + m2[index];
}

Matrix addMatrices(const Matrix& m1, const Matrix& m2) {
    DTYPE* result = allocate1DArrayDevice(m1.n * m1.m);

    addMatricesDevice<<<m1.n, m1.m>>>(m1.data, m2.data, result, m1.n, m1.m);

    return Matrix(result, m1.n, m1.m, DEVICE);
}

__global__
void subtractMatricesDevice(const DTYPE* m1, const DTYPE* m2, DTYPE* result, int n, int m) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n * m) {
        return;
    }

    result[index] = m1[index] - m2[index];
}

Matrix subtractMatrices(const Matrix& m1, const Matrix& m2) {
    DTYPE* result = allocate1DArrayDevice(m1.n * m1.m);

    subtractMatricesDevice<<<m1.n, m1.m>>>(m1.data, m2.data, result, m1.n, m1.m);

    return Matrix(result, m1.n, m1.m, DEVICE);
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

Vector multiplyMatrixVector(const Matrix& matrix, const Vector& vector) {
    DTYPE* result = allocate1DArrayDevice(matrix.n);

    mulMatrixVectorDevice<<<1, matrix.n>>>(matrix.data, vector.data, result, matrix.n, matrix.m);

    return Vector(result, matrix.n, DEVICE);
}

__global__
void multiplyMatrixDevice(const DTYPE* matrix, DTYPE constant, DTYPE* result, int n, int m) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n * m) {
        return;
    }

    result[index] = matrix[index] * constant;
}

Matrix multiplyMatrix(const Matrix& m1, DTYPE constant) {
    DTYPE* result = allocate1DArrayDevice(m1.n * m1.m);

    multiplyMatrixDevice<<<m1.n, m1.m>>>(m1.data, constant, result, m1.n, m1.m);

    return Matrix(result, m1.n, m1.m, DEVICE);
}
