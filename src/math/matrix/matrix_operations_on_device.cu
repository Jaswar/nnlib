///**
// * @file matrix_operations_on_device.cu
// * @brief Source file defining matrix operations that happen on device.
// * @author Jan Warchocki
// * @date 15 March 2022
// */
//
//#include "gpu/allocation_gpu.cuh"
//#include "gpu/assert.cuh"
//#include "matrix_operations_on_device.cuh"
//#include "verify.cuh"
//#include <algorithm>
//#include <exceptions/unexpected_cuda_call_exception.h>
//
//#ifdef HAS_CUDA
//
///**
// * @brief Size of a tile when performing tiled matrix multiplication.
// */
//#define TILE_WIDTH 16
//
//// NOLINTBEGIN(readability-static-accessed-through-instance)
//
///**
// * @brief Kernel method to add two matrices together.
// *
// * @param m1 The data of the first matrix.
// * @param m2 The data of the second matrix.
// * @param result Where the result of the operation should be stored.
// * @param n The number of rows in the operand matrices.
// * @param m The number of columns in the operand matrices.
// */
//__global__ void addMatricesKernel(const DTYPE* m1, const DTYPE* m2, DTYPE* result, size_t n, size_t m) {
//    auto index = blockIdx.x * blockDim.x + threadIdx.x;
//
//    if (index >= n * m) {
//        return;
//    }
//
//    result[index] = m1[index] + m2[index];
//}
//
///**
// * @brief Kernel method to perform broadcast-add operation.
// *
// * @param matrix The data of the matrix.
// * @param vector The data of the vector to broadcast.
// * @param result Where the result of the operation should be stored.
// * @param n The number of rows of the matrix.
// * @param m The number of columns of the matrix. Same as the size of the vector.
// */
//__global__ void addBroadcastKernel(const DTYPE* matrix, const DTYPE* vector, DTYPE* result, size_t n, size_t m) {
//    auto row = blockIdx.x;
//    auto column = threadIdx.x;
//
//    if (row > n || column > m) {
//        return;
//    }
//
//    result[row * m + column] = matrix[row * m + column] + vector[column];
//}
//
///**
// * @brief Kernel method to perform matrix subtraction.
// *
// * @param m1 The data of matrix to subtract from.
// * @param m2 The data of the matrix to be subtracted.
// * @param result Where the result of the operation should be stored.
// * @param n The number of rows of the matrices.
// * @param m The number of columns of the matrices.
// */
//__global__ void subtractMatricesKernel(const DTYPE* m1, const DTYPE* m2, DTYPE* result, size_t n, size_t m) {
//    auto index = blockIdx.x * blockDim.x + threadIdx.x;
//
//    if (index >= n * m) {
//        return;
//    }
//
//    result[index] = m1[index] - m2[index];
//}
//
///**
// * @brief Kernel method to perform matrix-vector multiplication.
// *
// * @param matrix The data of the matrix to multiply.
// * @param vector The data of the vector to multiply.
// * @param result Where the result of the operation should be stored.
// * @param n The number of rows of the matrix.
// * @param m The number of columns of the matrix. Same as the size of the vector.
// */
//__global__ void mulMatrixVectorKernel(const DTYPE* matrix, const DTYPE* vector, DTYPE* result, size_t n, size_t m) {
//    auto index = blockIdx.x * blockDim.x + threadIdx.x;
//
//    if (index >= n) {
//        return;
//    }
//
//    DTYPE sum = 0;
//    for (int i = 0; i < m; i++) {
//        sum += matrix[index * m + i] * vector[i];
//    }
//    result[index] = sum;
//}
//
///**
// * @brief Kernel method to multiply two matrices using a tiling method.
// *
// * This method currently is not used to perform matrix-matrix multiplication for performance reasons.
// *
// * @param m1 The data of the first matrix.
// * @param m2 The data of the second matrix.
// * @param result Where the result of the operation should be stored.
// * @param n The number of rows of the first matrix.
// * @param m The number of columns of the first matrix.
// * @param k The number of columns of the second matrix.
// */
//// NOLINTNEXTLINE(google-readability-function-size)
//__global__ void multiplyMatricesTilingKernel(const DTYPE* m1, const DTYPE* m2, DTYPE* result, size_t n, size_t m,
//                                             size_t k) {
//    __shared__ DTYPE m1Tile[TILE_WIDTH][TILE_WIDTH];
//    __shared__ DTYPE m2Tile[TILE_WIDTH][TILE_WIDTH];
//
//    auto row = blockIdx.y * blockDim.y + threadIdx.y;
//    auto column = blockIdx.x * blockDim.x + threadIdx.x;
//
//    DTYPE acc = 0;
//    // Ignore the downcast here.
//    // NOLINTNEXTLINE
//    for (int tileIdx = 0; tileIdx < std::ceil((float) m / TILE_WIDTH); tileIdx++) {
//        auto m1InxColumn = tileIdx * blockDim.x + threadIdx.x;
//        auto m2InxRow = tileIdx * blockDim.y + threadIdx.y;
//
//        if (row < n && m1InxColumn < m) {
//            m1Tile[threadIdx.y][threadIdx.x] = m1[row * m + m1InxColumn];
//        } else {
//            m1Tile[threadIdx.y][threadIdx.x] = 0;
//        }
//
//        if (m2InxRow < m && column < k) {
//            m2Tile[threadIdx.y][threadIdx.x] = m2[m2InxRow * k + column];
//        } else {
//            m2Tile[threadIdx.y][threadIdx.x] = 0;
//        }
//
//        __syncthreads();
//
//        for (int inx = 0; inx < TILE_WIDTH; inx++) {
//            acc += m1Tile[threadIdx.y][inx] * m2Tile[inx][threadIdx.x];
//        }
//
//        __syncthreads();
//    }
//
//    if (row < n && column < k) {
//        result[row * k + column] = acc;
//    }
//}
//
///**
// * @brief Kernel method to perform naive matrix-matrix multiplication.
// *
// * @param m1 The data of the first matrix.
// * @param m2 The data of the second matrix.
// * @param result Where the result of the operation should be stored.
// * @param n The number of rows of the first matrix.
// * @param m The number of columns of the first matrix.
// * @param k The number of columns of the second matrix.
// */
//__global__ void multiplyMatricesNoTilingKernel(const DTYPE* m1, const DTYPE* m2, DTYPE* result, size_t n, size_t m,
//                                               size_t k) {
//    auto index = blockIdx.x * blockDim.x + threadIdx.x;
//    auto row = index / k;
//    auto column = index % k;
//
//    if (row >= n || column >= k) {
//        return;
//    }
//
//    DTYPE sum = 0;
//    for (int i = 0; i < m; i++) {
//        sum += m1[row * m + i] * m2[i * k + column];
//    }
//
//    result[row * k + column] = sum;
//}
//
///**
// * @brief Kernel method to multiply a matrix with a constant.
// *
// * @param matrix The data of the matrix to multiply.
// * @param constant The constant to multiply @p matrix with.
// * @param result Where the result of the operation should be stored.
// * @param n The number of rows of the matrix.
// * @param m The number of columns of the matrix.
// */
//__global__ void multiplyMatrixKernel(const DTYPE* matrix, DTYPE constant, DTYPE* result, size_t n, size_t m) {
//    auto index = blockIdx.x * blockDim.x + threadIdx.x;
//
//    if (index >= n * m) {
//        return;
//    }
//
//    result[index] = matrix[index] * constant;
//}
//
///**
// * @brief Kernel method to apply hadamard product to two matrices.
// *
// * @param m1 The data of the first matrix.
// * @param m2 The data of the second matrix.
// * @param result Where the result of the hadamard operation should be stored.
// * @param n The number of rows of the matrices.
// * @param m The number of columns of the matrices.
// */
//__global__ void hadamardMatricesKernel(const DTYPE* m1, const DTYPE* m2, DTYPE* result, size_t n, size_t m) {
//    auto index = blockIdx.x * blockDim.x + threadIdx.x;
//
//    if (index >= n * m) {
//        return;
//    }
//
//    result[index] = m1[index] * m2[index];
//}
//
///**
// * @brief Kernel method to transpose a matrix.
// *
// * @param matrix The data of the matrix to transpose.
// * @param result Where the result of the transpose operation should be stored.
// * @param n The number of rows of @p matrix.
// * @param m The number of columns of @p matrix.
// */
//__global__ void transposeMatrixKernel(const DTYPE* matrix, DTYPE* result, size_t n, size_t m) {
//    auto row = blockIdx.x;
//    auto column = threadIdx.x;
//
//    if (row >= n || column >= m) {
//        return;
//    }
//
//    result[column * n + row] = matrix[row * m + column];
//}
//
//// NOLINTEND(readability-static-accessed-through-instance)
//
//void addMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result) {
//    addMatricesKernel<<<m1.n, m1.m>>>(m1.data, m2.data, result.data, m1.n, m1.m);
//    GPU_CHECK_ERROR(cudaGetLastError());
//}
//
//void addBroadcastOnDevice(const Matrix& m, const Vector& v, Matrix& result) {
//    addBroadcastKernel<<<m.n, m.m>>>(m.data, v.data, result.data, m.n, m.m);
//    GPU_CHECK_ERROR(cudaGetLastError());
//}
//
//void subtractMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result) {
//    subtractMatricesKernel<<<m1.n, m1.m>>>(m1.data, m2.data, result.data, m1.n, m1.m);
//    GPU_CHECK_ERROR(cudaGetLastError());
//}
//
//void multiplyMatrixVectorOnDevice(const Matrix& matrix, const Vector& vector, Vector& result) {
//    mulMatrixVectorKernel<<<1, matrix.n>>>(matrix.data, vector.data, result.data, matrix.n, matrix.m);
//    GPU_CHECK_ERROR(cudaGetLastError());
//}
//
//void multiplyMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result) {
//    // Linter says props will not be initialized, but it will be so disable error.
//    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
//    cudaDeviceProp props;
//    GPU_CHECK_ERROR(cudaGetDeviceProperties(&props, 0));
//    size_t threads = props.maxThreadsPerBlock;
//    multiplyMatricesNoTilingKernel<<<(m1.n * m2.m) / threads + 1, threads>>>(m1.data, m2.data, result.data, m1.n, m1.m,
//                                                                             m2.m);
//    GPU_CHECK_ERROR(cudaGetLastError());
//}
//
//void multiplyMatrixOnDevice(const Matrix& m, DTYPE constant, Matrix& result) {
//    multiplyMatrixKernel<<<m.n, m.m>>>(m.data, constant, result.data, m.n, m.m);
//    GPU_CHECK_ERROR(cudaGetLastError());
//}
//
//void hadamardMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result) {
//    hadamardMatricesKernel<<<m1.n, m1.m>>>(m1.data, m2.data, result.data, m1.n, m1.m);
//    GPU_CHECK_ERROR(cudaGetLastError());
//}
//
//void transposeMatrixOnDevice(const Matrix& m, Matrix& result) {
//    transposeMatrixKernel<<<m.n, m.m>>>(m.data, result.data, m.n, m.m);
//    GPU_CHECK_ERROR(cudaGetLastError());
//}
//
//#else
//
//void addMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result) {
//    throw UnexpectedCUDACallException();
//}
//
//void addBroadcastOnDevice(const Matrix& m, const Vector& v, Matrix& result) {
//    throw UnexpectedCUDACallException();
//}
//
//void subtractMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result) {
//    throw UnexpectedCUDACallException();
//}
//
//void multiplyMatrixVectorOnDevice(const Matrix& matrix, const Vector& vector, Vector& result) {
//    throw UnexpectedCUDACallException();
//}
//
//void multiplyMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result) {
//    throw UnexpectedCUDACallException();
//}
//
//void multiplyMatrixOnDevice(const Matrix& m, DTYPE constant, Matrix& result) {
//    throw UnexpectedCUDACallException();
//}
//
//void hadamardMatricesOnDevice(const Matrix& m1, const Matrix& m2, Matrix& result) {
//    throw UnexpectedCUDACallException();
//}
//
//void transposeMatrixOnDevice(const Matrix& m, Matrix& result) {
//    throw UnexpectedCUDACallException();
//}
//
//#endif
