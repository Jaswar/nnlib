/**
 * @file linear_activation_common.h
 * @brief 
 * @author Jan Warchocki
 * @date 11 February 2023
 */

#ifndef NNLIB_LINEAR_ACTIVATION_COMMON_H
#define NNLIB_LINEAR_ACTIVATION_COMMON_H

void linearActivationForwardPBT(bool testDevice);
void linearActivationComputeDerivativesPBT(bool testDevice);

#endif //NNLIB_LINEAR_ACTIVATION_COMMON_H
