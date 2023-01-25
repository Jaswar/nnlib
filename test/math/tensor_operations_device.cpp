/**
 * @file tensor_operations_device.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 13 October 2022
 */

#include <gtest/gtest.h>
#include <tensor.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include "tensor_operations_common.h"

#ifdef __CUDA__

RC_GTEST_PROP(tensor_operations_device, fill, (float value)) {
    tensorFillPBT(value, true);
}

RC_GTEST_PROP(tensor_operations_device, add, ()) {
    tensorAddPBT(true);
}

RC_GTEST_PROP(tensor_operations_device, add_broadcast, ()) {
    tensorAddBroadcastPBT(true);
}

RC_GTEST_PROP(tensor_operations_device, subtract, ()) {
    tensorSubtractPBT(true);
}

RC_GTEST_PROP(tensor_operations_device, hadamard, ()) {
    tensorHadamardPBT(true);
}

RC_GTEST_PROP(tensor_operations_device, divide, ()) {
    tensorDividePBT(true);
}

RC_GTEST_PROP(tensor_operations_device, log, ()) {
    tensorLogPBT(true);
}

RC_GTEST_PROP(tensor_operations_device, multiply_constant, (float constant)) {
    tensorMultiplyConstantPBT(constant, true);
}

RC_GTEST_PROP(tensor_operations_device, multiply_matrix_vector, ()) {
    tensorMultiplyMatrixVectorPBT(true);
}

RC_GTEST_PROP(tensor_operations_device, multiply_matrix_matrix, ()) {
    tensorMultiplyMatrixMatrixPBT(true);
}

RC_GTEST_PROP(tensor_operations_device, transpose, ()) {
    tensorTransposePBT(true);
}

#endif
