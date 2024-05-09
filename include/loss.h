/**
 * @file loss.h
 * @brief Header file declaring different loss functions.
 * @author Jan Warchocki
 * @date 23 December 2022
 */

#ifndef NNLIB_LOSS_H
#define NNLIB_LOSS_H

#include "metric.h"
#include "tensor.h"

/**
 * @brief Abstract class representing a loss function.
 *
 * All loss functions are by default also metrics.
 *
 * The child classes need to define the `calculateLoss` and `calculateDerivatives` methods.
 */
class Loss : public Metric {

    /**
     * @brief Constructor for the Loss class.
     *
     * Initializes the `numSamples` and `currentTotalLoss` variables to 0.
     */
public:
    Loss();

    /**
     * @brief Defines the method inherited from abstract Metric parent.
     *
     * It simply calls the #calculateLoss function, which is implemented by every child loss function.
     *
     * @param targets The desired outputs of the network.
     * @param predictions The actual outputs of the network.
     * @return The value of the metric. Here, the value of the loss function.
     */
    float calculateMetric(const sTensor& targets, const sTensor& predictions) override;

    virtual sTensor calculateLoss(const sTensor& targets, const sTensor& predictions) = 0;
};

/**
 * @brief Class representing the Mean Squared Error.
 */
class MeanSquaredError : public Loss {
public:
    sTensor calculateLoss(const sTensor& targets, const sTensor& predictions) override;

    [[nodiscard]] std::string getShortName() const override;
};

/**
 * @brief Class representing the Binary Cross Entropy.
 *
 * This loss expects the targets to be of shape (n, 1) with labels 0 or 1.
 */
class BinaryCrossEntropy : public Loss {
public:
    sTensor calculateLoss(const sTensor& targets, const sTensor& predictions) override;

    [[nodiscard]] std::string getShortName() const override;
};

/**
 * @brief Class representing the Categorical Cross Entropy.
 *
 * This loss expects the targets to be in shape (batchSize, numClasses), where each row contains only a single 1
 * and `numClasses - 1` 0s.
 *
 * The sum of predictions doesn't have to be 1. The sum will be auto-normalized when calculating the loss and
 * the derivatives.
 */
class CategoricalCrossEntropy : public Loss {
public:
    sTensor calculateLoss(const sTensor& targets, const sTensor& predictions) override;

    [[nodiscard]] std::string getShortName() const override;
};


#endif //NNLIB_LOSS_H
