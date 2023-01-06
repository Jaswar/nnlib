/**
 * @file tensor_operations_on_device.cu
 * @brief Source file defining tensor operations that happen on device.
 * @author Jan Warchocki
 * @date 29 August 2022
 */

#include "tensor_operations_on_device.cuh"
#include <exceptions/unexpected_cuda_call_exception.h>
#include <gpu/assert.cuh>
#include <cmath>

#ifdef __CUDA__

/**
 * @brief Size of a tile when performing tiled matrix multiplication.
 */
#define TILE_WIDTH 16

// NOLINTBEGIN(readability-static-accessed-through-instance)

/**
 * @brief Kernel method to add two tensors together.
 *
 * @param a The data of the first tensor.
 * @param b The data of the second tensor.
 * @param destination Where the result of the operation should be stored.
 * @param size The size of the tensors.
 */
__global__ void addTensorsKernel(const float* a, const float* b, float* destination, size_t size) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size) {
        return;
    }

    destination[index] = a[index] + b[index];
}

/**
 * @brief Kernel method to perform broadcast-add operation.
 *
 * @param matrix The data of the matrix.
 * @param vector The data of the vector to broadcast.
 * @param destination Where the result of the operation should be stored.
 * @param n The number of rows of the matrix.
 * @param m The number of columns of the matrix. Same as the size of the vector.
 */
__global__ void addBroadcastKernel(const float* matrix, const float* vector, float* destination, size_t n, size_t m) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    auto row = index / m;
    auto column = index % m;

    if (row > n || column > m) {
        return;
    }

    destination[row * m + column] = matrix[row * m + column] + vector[column];
}

/**
 * @brief Kernel method to perform tensor subtraction.
 *
 * @param a The data of the tensor to subtract from.
 * @param b The data of the tensor to be subtracted.
 * @param destination Where the result of the operation should be stored.
 * @param size The size of the tensors.
 */
__global__ void subtractTensorsKernel(const float* a, const float* b, float* destination, size_t size) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size) {
        return;
    }

    destination[index] = a[index] - b[index];
}

/**
 * @brief Kernel method to perform matrix-vector multiplication.
 *
 * @param matrix The data of the matrix to multiply.
 * @param vector The data of the vector to multiply.
 * @param destination Where the result of the operation should be stored.
 * @param n The number of rows of the matrix.
 * @param m The number of columns of the matrix. Same as the size of the vector.
 */
__global__ void mulMatrixVectorKernel(const float* matrix, const float* vector, float* destination, size_t n,
                                      size_t m) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    float sum = 0;
    for (int i = 0; i < m; i++) {
        sum += matrix[index * m + i] * vector[i];
    }
    destination[index] = sum;
}

/**
 * @brief Kernel method to multiply two matrices using a tiling method.
 *
 * This method currently is not used to perform matrix-matrix multiplication for performance reasons.
 *
 * @param m1 The data of the first matrix.
 * @param m2 The data of the second matrix.
 * @param destination Where the result of the operation should be stored.
 * @param n The number of rows of the first matrix.
 * @param m The number of columns of the first matrix.
 * @param k The number of columns of the second matrix.
 */
// NOLINTNEXTLINE(google-readability-function-size)
__global__ void multiplyMatricesTilingKernel(const float* m1, const float* m2, float* destination, size_t n, size_t m,
                                             size_t k) {
    __shared__ float m1Tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float m2Tile[TILE_WIDTH][TILE_WIDTH];

    auto row = blockIdx.y * blockDim.y + threadIdx.y;
    auto column = blockIdx.x * blockDim.x + threadIdx.x;

    float acc = 0;
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
        destination[row * k + column] = acc;
    }
}

/**
 * @brief Kernel method to perform naive matrix-matrix multiplication.
 *
 * @param m1 The data of the first matrix.
 * @param m2 The data of the second matrix.
 * @param destination Where the result of the operation should be stored.
 * @param n The number of rows of the first matrix.
 * @param m The number of columns of the first matrix.
 * @param k The number of columns of the second matrix.
 */
__global__ void multiplyMatricesNoTilingKernel(const float* m1, const float* m2, float* destination, size_t n, size_t m,
                                               size_t k) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    auto row = index / k;
    auto column = index % k;

    if (row >= n || column >= k) {
        return;
    }

    float sum = 0;
    for (int i = 0; i < m; i++) {
        sum += m1[row * m + i] * m2[i * k + column];
    }

    destination[row * k + column] = sum;
}

/**
 * @brief Kernel method to multiply a tensor with a constant.
 *
 * @param tensor The data of the tensor to multiply.
 * @param constant The constant to multiply @p tensor with.
 * @param destination Where the result of the operation should be stored.
 * @param size The size of the tensor.
 */
__global__ void multiplyTensorKernel(const float* tensor, float constant, float* destination, size_t size) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size) {
        return;
    }

    destination[index] = tensor[index] * constant;
}

/**
 * @brief Kernel method to apply hadamard product to two tensors.
 *
 * @param a The data of the first tensor.
 * @param b The data of the second tensor.
 * @param destination Where the result of the hadamard operation should be stored.
 * @param size The size of the tensors.
 */
__global__ void hadamardTensorsKernel(const float* a, const float* b, float* destination, size_t size) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size) {
        return;
    }

    destination[index] = a[index] * b[index];
}

__global__ void divideTensorsKernel(const float* a, const float* b, float* destination, size_t size) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size) {
        return;
    }

    destination[index] = a[index] / b[index];
}

__global__ void logTensorKernel(const float* a, float* destination, size_t size) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size) {
        return;
    }

    destination[index] = log(a[index]);
}

/**
 * @brief Kernel method to transpose a matrix.
 *
 * @param matrix The data of the matrix to transpose.
 * @param destination Where the result of the transpose operation should be stored.
 * @param n The number of rows of @p matrix.
 * @param m The number of columns of @p matrix.
 */
__global__ void transposeMatrixKernel(const float* matrix, float* destination, size_t n, size_t m) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    auto row = index / m;
    auto column = index % m;

    if (row >= n || column >= m) {
        return;
    }

    destination[column * n + row] = matrix[row * m + column];
}

__global__ void fillTensorKernel(float* tensor, float value, size_t size) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size) {
        return;
    }

    tensor[index] = value;
}

// NOLINTEND(readability-static-accessed-through-instance)

void addTensorsOnDevice(const Tensor& a, const Tensor& b, Tensor& destination) {
    auto grid = a.size / a.session.threadsPerBlock + 1;
    auto block = a.session.threadsPerBlock;
    addTensorsKernel<<<grid, block>>>(a.data, b.data, destination.data, a.size);
    GPU_CHECK_ERROR(cudaGetLastError());
}

void subtractTensorsOnDevice(const Tensor& a, const Tensor& b, Tensor& destination) {
    auto grid = a.size / a.session.threadsPerBlock + 1;
    auto block = a.session.threadsPerBlock;
    subtractTensorsKernel<<<grid, block>>>(a.data, b.data, destination.data, a.size);
    GPU_CHECK_ERROR(cudaGetLastError());
}

void hadamardTensorsOnDevice(const Tensor& a, const Tensor& b, Tensor& destination) {
    auto grid = a.size / a.session.threadsPerBlock + 1;
    auto block = a.session.threadsPerBlock;
    hadamardTensorsKernel<<<grid, block>>>(a.data, b.data, destination.data, a.size);
    GPU_CHECK_ERROR(cudaGetLastError());
}

void divideTensorsOnDevice(const Tensor& a, const Tensor& b, Tensor& destination) {
    auto grid = a.size / a.session.threadsPerBlock + 1;
    auto block = a.session.threadsPerBlock;
    divideTensorsKernel<<<grid, block>>>(a.data, b.data, destination.data, a.size);
    GPU_CHECK_ERROR(cudaGetLastError());
}

void logTensorOnDevice(const Tensor& a, Tensor& destination) {
    auto grid = a.size / a.session.threadsPerBlock + 1;
    auto block = a.session.threadsPerBlock;
    logTensorKernel<<<grid, block>>>(a.data, destination.data, a.size);
    GPU_CHECK_ERROR(cudaGetLastError());
}

void addBroadcastOnDevice(const Tensor& matrix, const Tensor& vector, Tensor& destination) {
    auto grid = matrix.size / matrix.session.threadsPerBlock + 1;
    auto block = matrix.session.threadsPerBlock;
    addBroadcastKernel<<<grid, block>>>(matrix.data, vector.data, destination.data, matrix.shape[0], matrix.shape[1]);
    GPU_CHECK_ERROR(cudaGetLastError());
}

void multiplyTensorOnDevice(const Tensor& tensor, float constant, Tensor& destination) {
    auto grid = tensor.size / tensor.session.threadsPerBlock + 1;
    auto block = tensor.session.threadsPerBlock;
    multiplyTensorKernel<<<grid, block>>>(tensor.data, constant, destination.data, tensor.size);
    GPU_CHECK_ERROR(cudaGetLastError());
}

void multiplyMatrixVectorOnDevice(const Tensor& matrix, const Tensor& vector, Tensor& destination) {
    auto grid = matrix.shape[0] / matrix.session.threadsPerBlock + 1;
    auto block = matrix.session.threadsPerBlock;
    mulMatrixVectorKernel<<<grid, block>>>(matrix.data, vector.data, destination.data, matrix.shape[0],
                                                  matrix.shape[1]);
    GPU_CHECK_ERROR(cudaGetLastError());
}

void multiplyMatrixMatrixOnDevice(const Tensor& m1, const Tensor& m2, Tensor& destination) {
    auto grid = m1.size / m1.session.threadsPerBlock + 1;
    auto block = m1.session.threadsPerBlock;
    multiplyMatricesNoTilingKernel<<<grid, block>>>(m1.data, m2.data, destination.data, m1.shape[0], m1.shape[1],
                                                    m2.shape[1]);
    GPU_CHECK_ERROR(cudaGetLastError());
}

void transposeMatrixOnDevice(const Tensor& matrix, Tensor& destination) {
    auto grid = matrix.size / matrix.session.threadsPerBlock + 1;
    auto block = matrix.session.threadsPerBlock;
    transposeMatrixKernel<<<grid, block>>>(matrix.data, destination.data, matrix.shape[0],
                                                                matrix.shape[1]);
    GPU_CHECK_ERROR(cudaGetLastError());
}

void fillTensorOnDevice(Tensor& tensor, float value) {
    auto grid = tensor.size / tensor.session.threadsPerBlock + 1;
    auto block = tensor.session.threadsPerBlock;
    fillTensorKernel<<<grid, block>>>(tensor.data, value, tensor.size);
    GPU_CHECK_ERROR(cudaGetLastError());
}

#else

void addTensorsOnDevice(const Tensor& a, const Tensor& b, Tensor& destination) {
    throw UnexpectedCUDACallException();
}

void subtractTensorsOnDevice(const Tensor& a, const Tensor& b, Tensor& destination) {
    throw UnexpectedCUDACallException();
}

void hadamardTensorsOnDevice(const Tensor& a, const Tensor& b, Tensor& destination) {
    throw UnexpectedCUDACallException();
}

void divideTensorsOnDevice(const Tensor& a, const Tensor& b, Tensor& destination) {
    throw UnexpectedCUDACallException();
}

void logTensorOnDevice(const Tensor& a, Tensor& destination) {
    throw UnexpectedCUDACallException();
}

void addBroadcastOnDevice(const Tensor& matrix, const Tensor& vector, Tensor& destination) {
    throw UnexpectedCUDACallException();
}

void multiplyTensorOnDevice(const Tensor& tensor, float constant, Tensor& destination) {
    throw UnexpectedCUDACallException();
}

void multiplyMatrixVectorOnDevice(const Tensor& matrix, const Tensor& vector, Tensor& destination) {
    throw UnexpectedCUDACallException();
}

void multiplyMatrixMatrixOnDevice(const Tensor& m1, const Tensor& m2, Tensor& destination) {
    throw UnexpectedCUDACallException();
}

void transposeMatrixOnDevice(const Tensor& matrix, Tensor& destination) {
    throw UnexpectedCUDACallException();
}

void fillTensorOnDevice(Tensor& tensor, float value) {
    throw UnexpectedCUDACallException();
}

#endif
