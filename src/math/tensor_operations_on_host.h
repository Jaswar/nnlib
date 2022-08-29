/**
 * @file tensor_operations_on_host.h
 * @brief 
 * @author Jan Warchocki
 * @date 29 August 2022
 */

#ifndef NNLIB_TENSOR_OPERATIONS_ON_HOST_H
#define NNLIB_TENSOR_OPERATIONS_ON_HOST_H

#include <tensor.h>

void addTensorsOnHost(const Tensor& a, const Tensor& b, Tensor& destination);
void subtractTensorsOnHost(const Tensor& a, const Tensor& b, Tensor& destination);
void hadamardTensorsOnHost(const Tensor& a, const Tensor& b, Tensor& destination);

void addBroadcastOnHost(const Tensor& matrix, const Tensor& vector, Tensor& destination);

void multiplyTensorOnHost(const Tensor& tensor, float constant, Tensor& destination);

void multiplyMatrixVectorOnHost(const Tensor& matrix, const Tensor& vector, Tensor& destination);
void multiplyMatrixMatrixOnHost(const Tensor& m1, const Tensor& m2, Tensor& destination);

void transposeMatrixOnHost(const Tensor& matrix, Tensor& destination);

#endif //NNLIB_TENSOR_OPERATIONS_ON_HOST_H
