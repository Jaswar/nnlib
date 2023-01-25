/**
 * @file tensor_operations_common.h
 * @brief 
 * @author Jan Warchocki
 * @date 25 January 2023
 */

#ifndef NNLIB_TENSOR_OPERATIONS_COMMON_H
#define NNLIB_TENSOR_OPERATIONS_COMMON_H

void tensorFillPBT(float value, bool testDevice);
void tensorAddPBT(bool testDevice);
void tensorAddBroadcastPBT(bool testDevice);
void tensorSubtractPBT(bool testDevice);
void tensorHadamardPBT(bool testDevice);
void tensorDividePBT(bool testDevice);
void tensorLogPBT(bool testDevice);
void tensorMultiplyConstantPBT(float constant, bool testDevice);
void tensorMultiplyMatrixVectorPBT(bool testDevice);
void tensorMultiplyMatrixMatrixPBT(bool testDevice);
void tensorTransposePBT(bool testDevice);

#endif //NNLIB_TENSOR_OPERATIONS_COMMON_H
