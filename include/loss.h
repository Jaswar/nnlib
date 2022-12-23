/**
 * @file loss.h
 * @brief 
 * @author Jan Warchocki
 * @date 23 December 2022
 */

#ifndef NNLIB_LOSS_H
#define NNLIB_LOSS_H

#include "tensor.h"

class Loss {
public:
    virtual float calculateLoss(const Tensor& targets, const Tensor& predictions) = 0;

    virtual void calculateDerivatives(const Tensor& targets, const Tensor& predictions, Tensor& destination) const = 0;
};

class MeanSquaredError : public Loss {
private:
    Tensor workingSpace;
public:
    float calculateLoss(const Tensor& targets, const Tensor& predictions) override;

    void calculateDerivatives(const Tensor& targets, const Tensor& predictions, Tensor& destination) const override;

};


#endif //NNLIB_LOSS_H
