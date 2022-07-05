//
// Created by Jan Warchocki on 15/03/2022.
//


#include "gpu/allocation_gpu.cuh"
#include "gpu/assert.cuh"
#include "matrix_operations_on_device.cuh"
#include "verify.cuh"
#include <algorithm>
#include <exceptions/unexpected_cuda_call_exception.h>

#ifdef HAS_CUDA

#define TILE_WIDTH 16

// NOLINTBEGIN(readability-static-accessed-through-instance)

__global__ void addMatricesKernel(const DTYPE* m1, const DTYPE* m2, DTYPE* result, size_t n, size_t m) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n * m) {
        return;
    }

    result[index] = m1[index] + m2[index];
}

__global__ void addBroadcastKernel(const DTYPE* matrix, const DTYPE* vector, DTYPE* result, size_t n, size_t m) {
    auto row = blockIdx.x;
    auto column = threadIdx.x;

    if (row > n || column > m) {
        return;
    }

    result[row * m + column] = matrix[row * m + column] + vector[column];
}

__global__ void subtractMatricesKernel(const DTYPE* m1, const DTYPE* m2, DTYPE* result, size_t n, size_t m) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n * m) {
        return;
    }

    result[index] = m1[index] - m2[index];
}

__global__ void mulMatrixVectorKernel(const DTYPE* matrix, const DTYPE* vector, DTYPE* result, size_t n, size_t m) {
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

// NOLINTNEXTLINE(google-readability-function-size)
__global__ void multiplyMatricesTilingKernel(const DTYPE* m1, const DTYPE* m2, DTYPE* result, size_t n, size_t m,
                                             size_t k) {
    __shared__ DTYPE m1Tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ DTYPE m2Tile[TILE_WIDTH][TILE_WIDTH];

    auto row = blockIdx.y * blockDim.y + threadIdx.y;
    auto column = blockIdx.x * blockDim.x + threadIdx.x;

    DTYPE acc = 0;
    // Ignore the downcast here.
    // NOLINTNEXTLINE
    for (int tileIdx = 0; tileIdx < std::ceil((float) m / TILE_WIDTH); tileIdx++) {
        auto m1InxColumn = tileIdx * blockDim.x + threadIdx.x;
        auto m2InxRow = tileIdx * blockDim.y + threadIdx.y;

        if (row < n && m1InxColumn < m) {
            m1Tile[threadIdx.y][threadIdx.x] = m1[row * m + m1InxColumn];
        } else {
            m1Tile[threadIdx.y][threadIdx.x] = 0;
        }

        if (m2InxRow < m && column < k) {
            m2Tile[threadIdx.y][threadIdx.x] = m2[m2InxRow * k + column];
        } else {
            m2Tile[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        for (int inx = 0; inx < TILE_WIDTH; inx++) {
            acc += m1Tile[threadIdx.y][inx] * m2Tile[inx][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && column < k) {
        result[row * k + column] = acc;
    }
}

__global__ void multiplyMatricesNoTilingKernel(const DTYPE* m1, const DTYPE* m2, DTYPE* result, size_t n, size_t m,
                                               size_t k) {
    auto row = blockIdx.x;
    auto column = threadIdx.x;

    if (row >= n || column >= k) {
        return;
    }

    DTYPE sum = 0;
    for (int i = 0; i < m; i++) {
        sum += m1[row * m + i] * m2[i * k + column];
    }

    result[row * k + column] = sum;
}

__global__ void multiplyMatrixKernel(const DTYPE* matrix, DTYPE constant, DTYPE* result, size_t n, size_t m) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n * m) {
        return;
    }

    result[index] = matrix[index] * constant;
}

__global__ void hadamardMatricesKernel(const DTYPE* m1, const DTYPE* m2, DTYPE* result, size_t n, size_t m) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n * m) {
        return;
    }

    result[index] = m1[index] * m2[index];
}

__global__ void transposeMatrixKernel(const DTYPE* matrix, DTYPE* result, size_t n, size_t m) {
    auto row = blockIdx.x;
    auto column = threadIdx.x;

    if (row >= n || column >= m) {
        return;
    }

    result[column * n + row] = matrix[row * m + column];
}

// NOLINTEND(readability-static-accessed-through-instance)

void addMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result) {
    addMatricesKernel<<<m1.n, m1.m>>>(m1.data, m2.data, result.data, m1.n, m1.m);
    GPU_CHECK_ERROR(cudaGetLastError());
    GPU_CHECK_ERROR(cudaDeviceSynchronize());
}

void addBroadcastOnDevice(const Matrix& m, const Vector& v, Matrix& result) {
    addBroadcastKernel<<<m.n, m.m>>>(m.data, v.data, result.data, m.n, m.m);
    GPU_CHECK_ERROR(cudaGetLastError());
    GPU_CHECK_ERROR(cudaDeviceSynchronize());
}

void subtractMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result) {
    subtractMatricesKernel<<<m1.n, m1.m>>>(m1.data, m2.data, result.data, m1.n, m1.m);
    GPU_CHECK_ERROR(cudaGetLastError());
    GPU_CHECK_ERROR(cudaDeviceSynchronize());
}

void multiplyMatrixVectorOnDevice(const Matrix& matrix, const Vector& vector, Vector& result) {
    mulMatrixVectorKernel<<<1, matrix.n>>>(matrix.data, vector.data, result.data, matrix.n, matrix.m);
    GPU_CHECK_ERROR(cudaGetLastError());
    GPU_CHECK_ERROR(cudaDeviceSynchronize());
}

void multiplyMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result) {
    // Linter says props will not be initialized, but it will be so disable error.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    cudaDeviceProp props;
    GPU_CHECK_ERROR(cudaGetDeviceProperties(&props, 0));
    if (m1.m > props.maxThreadsPerBlock) {
        size_t sizeY = std::ceil(static_cast<double>(std::max(m1.n, m2.n)) / TILE_WIDTH);
        size_t sizeX = std::ceil(static_cast<double>(std::max(m1.m, m2.m)) / TILE_WIDTH);

        dim3 blocks(sizeX, sizeY);
        dim3 threads(TILE_WIDTH, TILE_WIDTH);
        multiplyMatricesTilingKernel<<<blocks, threads>>>(m1.data, m2.data, result.data, m1.n, m1.m, m2.m);
    } else {
        multiplyMatricesNoTilingKernel<<<m1.n, m2.m>>>(m1.data, m2.data, result.data, m1.n, m1.m, m2.m);
    }
    GPU_CHECK_ERROR(cudaGetLastError());
    GPU_CHECK_ERROR(cudaDeviceSynchronize());
}

void multiplyMatrixOnDevice(const Matrix& m1, DTYPE constant, Matrix& result) {
    multiplyMatrixKernel<<<m1.n, m1.m>>>(m1.data, constant, result.data, m1.n, m1.m);
    GPU_CHECK_ERROR(cudaGetLastError());
    GPU_CHECK_ERROR(cudaDeviceSynchronize());
}

void hadamardMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result) {
    hadamardMatricesKernel<<<m1.n, m1.m>>>(m1.data, m2.data, result.data, m1.n, m1.m);
    GPU_CHECK_ERROR(cudaGetLastError());
    GPU_CHECK_ERROR(cudaDeviceSynchronize());
}

void transposeMatrixOnDevice(const Matrix& m, Matrix& result) {
    transposeMatrixKernel<<<m.n, m.m>>>(m.data, result.data, m.n, m.m);
    GPU_CHECK_ERROR(cudaGetLastError());
    GPU_CHECK_ERROR(cudaDeviceSynchronize());
}

#else

void addMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result) {
    throw UnexpectedCUDACallException();
}

void addBroadcastOnDevice(const Matrix& m, const Vector& v, Matrix& result) {
    throw UnexpectedCUDACallException();
}

void subtractMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result) {
    throw UnexpectedCUDACallException();
}

void multiplyMatrixVectorOnDevice(const Matrix& matrix, const Vector& vector, Vector& result) {
    throw UnexpectedCUDACallException();
}

void multiplyMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result) {
    throw UnexpectedCUDACallException();
}

void multiplyMatrixOnDevice(const Matrix& m1, DTYPE constant, Matrix& result) {
    throw UnexpectedCUDACallException();
}

void hadamardMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result) {
    throw UnexpectedCUDACallException();
}

void transposeMatrixOnDevice(const Matrix& m, Matrix& result) {
    throw UnexpectedCUDACallException();
}

#endif
