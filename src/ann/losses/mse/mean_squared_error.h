/**
 * @file mean_squared_error.h
 * @brief 
 * @author Jan Warchocki
 * @date 23 December 2022
 */

#ifndef NNLIB_MEAN_SQUARED_ERROR_H
#define NNLIB_MEAN_SQUARED_ERROR_H

#include "ann/losses/loss.h"

class MeanSquaredError : public Loss {
public:
    float calculateLoss(const Tensor& targets, const Tensor& predictions) const override;

    void calculateDerivatives(const Tensor& targets, const Tensor& predictions, Tensor& destination) const override;
};


#endif //NNLIB_MEAN_SQUARED_ERROR_H
