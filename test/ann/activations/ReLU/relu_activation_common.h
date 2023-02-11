/**
 * @file relu_activation_common.h
 * @brief 
 * @author Jan Warchocki
 * @date 11 February 2023
 */

#ifndef NNLIB_RELU_ACTIVATION_COMMON_H
#define NNLIB_RELU_ACTIVATION_COMMON_H

void reluActivationForwardPBT(bool testDevice);
void reluActivationComputeDerivativesPBT(bool testDevice);

#endif //NNLIB_RELU_ACTIVATION_COMMON_H
