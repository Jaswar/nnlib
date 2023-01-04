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
protected:
    uint64_t numSamples;
    float currentTotalLoss;

public:
    Loss();

    void reset();

    virtual float calculateLoss(const Tensor& targets, const Tensor& predictions) = 0;

    virtual void calculateDerivatives(const Tensor& targets, const Tensor& predictions, Tensor& destination) = 0;
};

class MeanSquaredError : public Loss {
private:
    Tensor workingSpace;

public:
    float calculateLoss(const Tensor& targets, const Tensor& predictions) override;

    void calculateDerivatives(const Tensor& targets, const Tensor& predictions, Tensor& destination) override;
};

class BinaryCrossEntropy : public Loss {
private:
    Tensor ones;
    Tensor workingSpace;
    Tensor workingSpace2;

public:
    float calculateLoss(const Tensor& targets, const Tensor& predictions) override;

    void calculateDerivatives(const Tensor& targets, const Tensor& predictions, Tensor& destination) override;

private:
    void allocateOnes(const Tensor& targets, const Tensor& predictions);
    void allocateWorkingSpaces(const Tensor& targets, const Tensor& predictions);
};

class CategoricalCrossEntropy : public Loss {
private:
    Tensor workingSpace;

    Tensor ones;
    Tensor accumulatedSums;

public:
    float calculateLoss(const Tensor& targets, const Tensor& predictions) override;

    void calculateDerivatives(const Tensor& targets, const Tensor& predictions, Tensor& destination) override;

private:
    void allocateWorkingSpace(const Tensor& targets, const Tensor& predictions);

    void allocateOnes(const Tensor& targets, const Tensor& predictions);
    void allocateAccumulatedSums(const Tensor& targets, const Tensor& predictions);
};


#endif //NNLIB_LOSS_H
