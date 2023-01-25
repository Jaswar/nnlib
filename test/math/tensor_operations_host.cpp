//
// Created by Jan Warchocki on 26/08/2022.
//

#include <gtest/gtest.h>
#include <tensor.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include "tensor_operations_common.h"


RC_GTEST_PROP(tensor_operations_host, fill, (float value)) {
    tensorFillPBT(value, false);
}

RC_GTEST_PROP(tensor_operations_host, add, ()) {
    tensorAddPBT(false);
}

RC_GTEST_PROP(tensor_operations_host, add_broadcast, ()) {
    tensorAddBroadcastPBT(false);
}

RC_GTEST_PROP(tensor_operations_host, subtract, ()) {
    tensorSubtractPBT(false);
}

RC_GTEST_PROP(tensor_operations_host, hadamard, ()) {
    tensorHadamardPBT(false);
}

RC_GTEST_PROP(tensor_operations_host, divide, ()) {
    tensorDividePBT(false);
}

RC_GTEST_PROP(tensor_operations_host, log, ()) {
    tensorLogPBT(false);
}

RC_GTEST_PROP(tensor_operations_host, multiply_constant, (float constant)) {
    tensorMultiplyConstantPBT(constant, false);
}

RC_GTEST_PROP(tensor_operations_host, multiply_matrix_vector, ()) {
    tensorMultiplyMatrixVectorPBT(false);
}

RC_GTEST_PROP(tensor_operations_host, multiply_matrix_matrix, ()) {
    tensorMultiplyMatrixMatrixPBT(false);
}

RC_GTEST_PROP(tensor_operations_host, transpose, ()) {
    tensorTransposePBT(false);
}
