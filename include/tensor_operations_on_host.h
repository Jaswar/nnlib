/**
 * @file tensor_operations_on_host.h
 * @brief Header file declaring tensor operations that happen on host.
 *
 * The methods declared in this file are only called when all operands are located on the CPU.
 *
 * These methods do not perform any checking with regards to the size or location of the operands. This is already
 * done in the corresponding methods in tensor.cpp.
 * @author Jan Warchocki
 * @date 29 August 2022
 */

#ifndef NNLIB_TENSOR_OPERATIONS_ON_HOST_H
#define NNLIB_TENSOR_OPERATIONS_ON_HOST_H

#include "tensor.h"

/**
 * @brief Sum all values of a tensor.
 *
 * @param tensor The tensor to sum.
 * @return The sum of all values of a tensor.
 */
void sumTensorOnHost(const Tensor<float>& tensor, Tensor<float>& destination);

/**
 * @brief Fill a tensor with a constant value.
 *
 * @param tensor The tensor to fill.
 * @param value The value to fill the tensor with.
 */
void fillTensorOnHost(Tensor<float>& tensor, float value);

void fillTensorOnHost(Tensor<float>& tensor, const Tensor<float>& value);

/**
 * @brief Element-wise add two tensors.
 *
 * @param a The first tensor.
 * @param b The second tensor.
 * @param destination Where the result of the addition should be stored.
 */
void addTensorsOnHost(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& destination);

/**
 * @brief Subtract one tensor from another.
 *
 * @param a The tensor to subtract from.
 * @param b The tensor to be subtracted.
 * @param destination Where the result of the subtraction should be stored.
 */
void subtractTensorsOnHost(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& destination);

/**
 * @brief Perform hadamard product (element-wise multiplication) between two tensors.
 *
 * @param a The first tensor.
 * @param b The second tensor.
 * @param destination Where the result of the operation should be stored.
 */
void hadamardTensorsOnHost(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& destination);

/**
 * @brief Divide one tensor by another.
 *
 * @param a The tensor to divide.
 * @param b The tensor to divide by.
 * @param destination Where the result of the operation should be stored.
 */
void divideTensorsOnHost(const Tensor<float>& a, const Tensor<float>& b, Tensor<float>& destination);

/**
 * @brief Apply natural logarithm to each element of the tensor.
 *
 * @param a The tensor to apply natural logarithm to.
 * @param destination Where the result of the operation should be stored.
 */
void logTensorOnHost(const Tensor<float>& a, Tensor<float>& destination);

/**
 * @brief Perform the broadcast-add operation.
 *
 * @param matrix The matrix tensor.
 * @param vector The vector tensor.
 * @param destination Where the result of the addition should be stored.
 */
void addBroadcastOnHost(const Tensor<float>& matrix, const Tensor<float>& vector, Tensor<float>& destination);

/**
 * @brief Multiply a tensor with a constant.
 *
 * @param tensor The tensor to multiply.
 * @param constant The constant to multiply with.
 * @param destination Where the result of the multiplication should be stored.
 */
void multiplyTensorOnHost(const Tensor<float>& tensor, float constant, Tensor<float>& destination);

/**
 * @brief Multiply a matrix with a vector.
 *
 * @param matrix The matrix tensor.
 * @param vector The vector tensor.
 * @param destination Where the result of the multiplication should be stored.
 */
void multiplyMatrixVectorOnHost(const Tensor<float>& matrix, const Tensor<float>& vector, Tensor<float>& destination);

/**
 * @brief Multiply a matrix with a matrix.
 *
 * @param m1 The first matrix tensor.
 * @param m2 The second matrix tensor.
 * @param destination Where the result of the multiplication should be stored.
 */
void multiplyMatrixMatrixOnHost(const Tensor<float>& m1, const Tensor<float>& m2, Tensor<float>& destination);

/**
 * @brief Transpose a matrix.
 *
 * @param matrix The matrix vector to transpose.
 * @param destination Where the result of the transpose operation should be stored.
 */
void transposeMatrixOnHost(const Tensor<float>& matrix, Tensor<float>& destination);

void reluTensorOnHost(const Tensor<float>& tensor, Tensor<float>& destination);
void reluDerivativeTensorOnHost(const Tensor<float>& tensor, Tensor<float>& destination);

void sigmoidTensorOnHost(const Tensor<float>& tensor, Tensor<float>& destination);

#endif //NNLIB_TENSOR_OPERATIONS_ON_HOST_H
