/**
 * @file sigmoid_activation_common.h
 * @brief 
 * @author Jan Warchocki
 * @date 11 February 2023
 */

#ifndef NNLIB_SIGMOID_ACTIVATION_COMMON_H
#define NNLIB_SIGMOID_ACTIVATION_COMMON_H

void sigmoidActivationForwardPBT(bool testDevice);
void sigmoidActivationComputeDerivativesPBT(bool testDevice);

#endif //NNLIB_SIGMOID_ACTIVATION_COMMON_H
