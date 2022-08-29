/**
 * @file tensor_operations_on_device.cuh
 * @brief 
 * @author Jan Warchocki
 * @date 29 August 2022
 */

#ifndef NNLIB_TENSOR_OPERATIONS_ON_DEVICE_CUH
#define NNLIB_TENSOR_OPERATIONS_ON_DEVICE_CUH

#include <tensor.h>

void addTensorsOnDevice(const Tensor& a, const Tensor& b, Tensor& destination);
void subtractTensorsOnDevice(const Tensor& a, const Tensor& b, Tensor& destination);
void hadamardTensorsOnDevice(const Tensor& a, const Tensor& b, Tensor& destination);

void addBroadcastOnDevice(const Tensor& matrix, const Tensor& vector, Tensor& destination);

void multiplyTensorOnDevice(const Tensor& tensor, float constant, Tensor& destination);

void multiplyMatrixVectorOnDevice(const Tensor& matrix, const Tensor& vector, Tensor& destination);
void multiplyMatrixMatrixOnDevice(const Tensor& m1, const Tensor& m2, Tensor& destination);

void transposeMatrixOnDevice(const Tensor& matrix, Tensor& destination);

#endif //NNLIB_TENSOR_OPERATIONS_ON_DEVICE_CUH
