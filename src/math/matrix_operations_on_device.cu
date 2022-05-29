//
// Created by Jan Warchocki on 15/03/2022.
//


#include <algorithm>
#include "matrix_operations_on_device.cuh"
#include "../gpu/allocation_gpu.cuh"
#include "verify.cuh"
#include "../gpu/assert.cuh"

#define TILE_WIDTH 16

__global__
void addMatricesKernel(const DTYPE* m1, const DTYPE* m2, DTYPE* result, size_t n, size_t m) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n * m) {
        return;
    }

    result[index] = m1[index] + m2[index];
}

void addMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result) {
    addMatricesKernel<<<m1.n, m1.m>>>(m1.data, m2.data, result.data, m1.n, m1.m);
    gpuCheckError( cudaGetLastError() )
    gpuCheckError( cudaDeviceSynchronize() )
}

__global__
void addBroadcastKernel(const DTYPE* matrix, const DTYPE* vector, DTYPE* result, size_t n, size_t m) {
    auto row = blockIdx.x;
    auto column = threadIdx.x;

    if (row > n || column > m) {
        return;
    }

    result[row * m + column] = matrix[row * m + column] + vector[column];
}

void addBroadcastOnDevice(const Matrix& m, const Vector& v, Matrix& result) {
    addBroadcastKernel<<<m.n, m.m>>>(m.data, v.data, result.data, m.n, m.m);
    gpuCheckError( cudaGetLastError() )
    gpuCheckError( cudaDeviceSynchronize() )
}

__global__
void subtractMatricesKernel(const DTYPE* m1, const DTYPE* m2, DTYPE* result, size_t n, size_t m) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n * m) {
        return;
    }

    result[index] = m1[index] - m2[index];
}

void subtractMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result) {
    subtractMatricesKernel<<<m1.n, m1.m>>>(m1.data, m2.data, result.data, m1.n, m1.m);
    gpuCheckError( cudaGetLastError() )
    gpuCheckError( cudaDeviceSynchronize() )
}

__global__
void mulMatrixVectorKernel(const DTYPE* matrix, const DTYPE* vector, DTYPE* result, size_t n, size_t m) {
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

void multiplyMatrixVectorOnDevice(const Matrix& matrix, const Vector& vector, Vector& result) {
    mulMatrixVectorKernel<<<1, matrix.n>>>(matrix.data, vector.data, result.data, matrix.n, matrix.m);
    gpuCheckError( cudaGetLastError() )
    gpuCheckError( cudaDeviceSynchronize() )
}

__global__
void multiplyMatricesTilingKernel(const DTYPE* m1, const DTYPE* m2, DTYPE* result, size_t N, size_t M, size_t K) {
    __shared__ DTYPE m1Tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ DTYPE m2Tile[TILE_WIDTH][TILE_WIDTH];

    auto row = blockIdx.y * blockDim.y + threadIdx.y;
    auto column = blockIdx.x * blockDim.x + threadIdx.x;

    DTYPE acc = 0;
    for (int tileIdx = 0; tileIdx < std::ceil((float) M / TILE_WIDTH); tileIdx++) {
        auto m1InxColumn = tileIdx * blockDim.x + threadIdx.x;
        auto m2InxRow = tileIdx * blockDim.y + threadIdx.y;

        if (row < N && m1InxColumn < M) {
            m1Tile[threadIdx.y][threadIdx.x] = m1[row * M + m1InxColumn];
        } else {
            m1Tile[threadIdx.y][threadIdx.x] = 0;
        }

        if (m2InxRow < M && column < K) {
            m2Tile[threadIdx.y][threadIdx.x] = m2[m2InxRow * K + column];
        } else {
            m2Tile[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            acc += m1Tile[threadIdx.y][k] * m2Tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && column < K) {
        result[row * K + column] = acc;
    }
}

__global__
void multiplyMatricesNoTilingKernel(const DTYPE* m1, const DTYPE* m2, DTYPE* result, size_t N, size_t M, size_t K) {
    auto row = blockIdx.x;
    auto column = threadIdx.x;

    if (row >= N || column >= K) {
        return;
    }

    DTYPE sum = 0;
    for (int i = 0; i < M; i++) {
        sum += m1[row * M + i] * m2[i * K + column];
    }

    result[row * K + column] = sum;
}

void multiplyMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result) {
    cudaDeviceProp props;
    gpuCheckError( cudaGetDeviceProperties(&props, 0) )
    if (m1.m > props.maxThreadsPerBlock) {
        size_t sizeY = std::ceil((double) std::max(m1.n, m2.n) / TILE_WIDTH);
        size_t sizeX = std::ceil((double) std::max(m1.m, m2.m) / TILE_WIDTH);

        dim3 blocks(sizeX, sizeY);
        dim3 threads(TILE_WIDTH, TILE_WIDTH);
        multiplyMatricesTilingKernel<<<blocks, threads>>>(m1.data, m2.data, result.data, m1.n, m1.m, m2.m);
    } else {
        multiplyMatricesNoTilingKernel<<<m1.n, m2.m>>>(m1.data, m2.data, result.data, m1.n, m1.m, m2.m);
    }
    gpuCheckError( cudaGetLastError() )
    gpuCheckError( cudaDeviceSynchronize() )
}

__global__
void multiplyMatrixKernel(const DTYPE* matrix, DTYPE constant, DTYPE* result, size_t n, size_t m) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n * m) {
        return;
    }

    result[index] = matrix[index] * constant;
}

void multiplyMatrixOnDevice(const Matrix& m1, DTYPE constant, Matrix& result) {
    multiplyMatrixKernel<<<m1.n, m1.m>>>(m1.data, constant, result.data, m1.n, m1.m);
    gpuCheckError( cudaGetLastError() )
    gpuCheckError( cudaDeviceSynchronize() )
}

__global__
void hadamardMatricesKernel(const DTYPE* m1, const DTYPE* m2, DTYPE* result, size_t n, size_t m) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n * m) {
        return;
    }

    result[index] = m1[index] * m2[index];
}

void hadamardMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result) {
    hadamardMatricesKernel<<<m1.n, m1.m>>>(m1.data, m2.data, result.data, m1.n, m1.m);
    gpuCheckError( cudaGetLastError() )
    gpuCheckError( cudaDeviceSynchronize() )
}

__global__
void transposeMatrixKernel(const DTYPE* matrix, DTYPE* result, size_t n, size_t m) {
    auto row = blockIdx.x;
    auto column = threadIdx.x;

    if (row >= n || column >= m) {
        return;
    }

    result[column * n + row] = matrix[row * m + column];
}

void transposeMatrixOnDevice(const Matrix& m, Matrix& result) {
    transposeMatrixKernel<<<m.n, m.m>>>(m.data, result.data, m.n, m.m);
    gpuCheckError( cudaGetLastError() )
    gpuCheckError( cudaDeviceSynchronize() )
}
