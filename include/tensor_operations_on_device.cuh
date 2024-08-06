/**
 * @file tensor_operations_on_device.cuh
 * @brief Header file declaring tensor operations that happen on device.
 *
 * The methods declared in this file are only called when all operands are located on the GPU.
 *
 * These methods do not perform any checking with regards to the size or location of the operands. This is already
 * done in the corresponding methods in tensor.cpp.
 * @author Jan Warchocki
 * @date 29 August 2022
 */

#ifndef NNLIB_TENSOR_OPERATIONS_ON_DEVICE_CUH
#define NNLIB_TENSOR_OPERATIONS_ON_DEVICE_CUH

#include "tensor.h"

void sumTensorOnDevice(const Tensor<float>& tensor, Tensor<float>& destination);

/** @copydoc tensor_operations_on_host.h::fillTensorOnHost() */
void fillTensorOnDevice(Tensor<float>& tensor, float value);

void fillTensorOnDevice(Tensor<float>& tensor, const Tensor<float>& value);

/** @copydoc tensor_operations_on_host.h::addTensorsOnHost() */
void addTensorsOnDevice(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& destination);

/** @copydoc tensor_operations_on_host.h::subtractTensorsOnHost() */
void subtractTensorsOnDevice(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& destination);

/** @copydoc tensor_operations_on_host.h::hadamardTensorsOnHost() */
void hadamardTensorsOnDevice(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& destination);

/** @copydoc tensor_operations_on_host.h::divideTensorsOnHost() */
void divideTensorsOnDevice(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& destination);

/** @copydoc tensor_operations_on_host.h::logTensorOnHost() */
void logTensorOnDevice(const Tensor<float>& a, Tensor<float>& destination);

/** @copydoc tensor_operations_on_host.h::addBroadcastOnHost() */
void addBroadcastOnDevice(const Tensor<float>& matrix, const Tensor<float>& vector, Tensor<float>& destination);

/** @copydoc tensor_operations_on_host.h::multiplyTensorOnHost() */
void multiplyTensorOnDevice(const Tensor<float>& tensor, float constant, Tensor<float>& destination);

/** @copydoc tensor_operations_on_host.h::multiplyMatrixVectorOnHost() */
void multiplyMatrixVectorOnDevice(const Tensor<float>& matrix, const Tensor<float>& vector, Tensor<float>& destination);

/** @copydoc tensor_operations_on_host.h::multiplyMatrixMatrixOnHost() */
void multiplyMatrixMatrixOnDevice(const Tensor<float>& m1, const Tensor<float>& m2, Tensor<float>& destination);

/** @copydoc tensor_operations_on_host.h::transposeMatrixOnHost() */
void transposeMatrixOnDevice(const Tensor<float>& matrix, Tensor<float>& destination);

void reluTensorOnDevice(const Tensor<float>& tensor, Tensor<float>& destination);
void reluDerivativeTensorOnDevice(const Tensor<float>& tensor, Tensor<float>& destination);

void sigmoidTensorOnDevice(const Tensor<float>& tensor, Tensor<float>& destination);

void castTensorOnDevice(const Tensor<float>& tensor, Tensor<double>& destination);
void castTensorOnDevice(const Tensor<double>& tensor, Tensor<float>& destination);

#ifdef __CUDA__

template<typename S, typename D>
__global__ void castTensorKernel(const S* tensor, D* destination, size_t size) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size) {
        return;
    }

    destination[index] = static_cast<D>(tensor[index]);
}

template<typename S, typename D>
void castTensorOnDevice(const Tensor<S>& tensor, Tensor<D>& destination) {
    auto grid = tensor.size / tensor.session.threadsPerBlock + 1;
    auto block = tensor.session.threadsPerBlock;
    castTensorKernel<<<grid, block>>>(tensor.data, destination.data, tensor.size);
    GPU_CHECK_ERROR(cudaGetLastError());
}

#else

template<typename S, typename D>
void castTensorOnDevice(const Tensor<S>& tensor, Tensor<D>& destination) {
    throw UnexpectedCUDACallException();
}

#endif

#endif //NNLIB_TENSOR_OPERATIONS_ON_DEVICE_CUH
