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

#include <tensor.h>

/** @copydoc tensor_operations_on_host.h::fillTensorOnHost() */
void fillTensorOnDevice(Tensor& tensor, float value);

/** @copydoc tensor_operations_on_host.h::addTensorsOnHost() */
void addTensorsOnDevice(const Tensor& a, const Tensor& b, Tensor& destination);

/** @copydoc tensor_operations_on_host.h::subtractTensorsOnHost() */
void subtractTensorsOnDevice(const Tensor& a, const Tensor& b, Tensor& destination);

/** @copydoc tensor_operations_on_host.h::hadamardTensorsOnHost() */
void hadamardTensorsOnDevice(const Tensor& a, const Tensor& b, Tensor& destination);

/** @copydoc tensor_operations_on_host.h::addBroadcastOnHost() */
void addBroadcastOnDevice(const Tensor& matrix, const Tensor& vector, Tensor& destination);

/** @copydoc tensor_operations_on_host.h::multiplyTensorOnHost() */
void multiplyTensorOnDevice(const Tensor& tensor, float constant, Tensor& destination);

/** @copydoc tensor_operations_on_host.h::multiplyMatrixVectorOnHost() */
void multiplyMatrixVectorOnDevice(const Tensor& matrix, const Tensor& vector, Tensor& destination);

/** @copydoc tensor_operations_on_host.h::multiplyMatrixMatrixOnHost() */
void multiplyMatrixMatrixOnDevice(const Tensor& m1, const Tensor& m2, Tensor& destination);

/** @copydoc tensor_operations_on_host.h::transposeMatrixOnHost() */
void transposeMatrixOnDevice(const Tensor& matrix, Tensor& destination);

#endif //NNLIB_TENSOR_OPERATIONS_ON_DEVICE_CUH
