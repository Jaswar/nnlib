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
    float calculateMetric(const Tensor& targets, const Tensor& predictions) override;

    /**
     * @brief Calculates the average loss so far in an epoch.
     *
     * After each batch, the loss in that batch is computed and added to the total. The size of the batch is then added
     * to `numSamples`. The returned value should then be `currentTotalLoss` divided by `numSamples` to get the average.
     *
     * @param targets The expected output of the network.
     * @param predictions The actual output of the network.
     * @return The average loss so far in an epoch.
     */
    virtual float calculateLoss(const Tensor& targets, const Tensor& predictions) = 0;

    /**
     * @brief Computes the partial derivative of the loss with respect to the prediction.
     *
     * @param targets The expected output of the network.
     * @param predictions The actual output of the network.
     * @param destination The Tensor where the derivatives should be saved.
     */
    virtual void calculateDerivatives(const Tensor& targets, const Tensor& predictions, Tensor& destination) = 0;
};

/**
 * @brief Class representing the Mean Squared Error.
 */
class MeanSquaredError : public Loss {
    /**
     * @brief Space used for computation of the loss/derivatives.
     */
private:
    Tensor workingSpace;

    /**
     * @copybrief Loss::calculateLoss
     *
     * The loss is calculated for each data sample as @f$ \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2 @f$
     * with `n` being the number of outputs.
     *
     * @copydetails Loss::calculateLoss
     */
public:
    float calculateLoss(const Tensor& targets, const Tensor& predictions) override;

    /**
     * @copybrief Loss::calculateDerivatives
     *
     * The derivative is calculated for each data sample as @f$ \frac{2}{n} (\hat{y}_i - y_i) @f$
     * with `n` being the number of outputs.
     *
     * @copydetails Loss::calculateDerivatives
     */
    void calculateDerivatives(const Tensor& targets, const Tensor& predictions, Tensor& destination) override;

    std::string getShortName() const override;
};

/**
 * @brief Class representing the Binary Cross Entropy.
 *
 * This loss expects the targets to be of shape (n, 1) with labels 0 or 1.
 */
class BinaryCrossEntropy : public Loss {

    /**
     * @brief Space containing only ones. Used when calculating the loss.
     */
private:
    Tensor onesLoss;

    /**
     * @brief Space containing only ones. Used when calculating the derivatives.
     */
    Tensor onesDerivatives;

    /**
     * @brief Space used for computation of the loss.
     */
    Tensor workingSpace;

    /**
     * @brief Space used for computation of the loss.
     */
    Tensor workingSpace2;

    /**
     * @brief Space used for computation of the derivatives.
     */
    Tensor workingSpace3;

    /**
     * @copybrief Loss::calculateLoss
     *
     * The loss is calculated for each data sample as @f$ -(y \ln(\hat{y}) + (1 - y)\ln(1 - \hat{y})) @f$.
     *
     * @copydetails Loss::calculateLoss
     */
public:
    float calculateLoss(const Tensor& targets, const Tensor& predictions) override;

    /**
     * @copybrief Loss::calculateDerivatives
     *
     * The derivative is calculated for each data sample as @f$ \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})} @f$.
     *
     * @copydetails Loss::calculateDerivatives
     */
    void calculateDerivatives(const Tensor& targets, const Tensor& predictions, Tensor& destination) override;

    /**
     * @brief Helper method to allocate the working spaces for derivatives.
     *
     * @param targets The expected output of the network.
     * @param predictions The actual output of the network.
     */
private:
    void allocateWorkingSpacesDerivatives(const Tensor& targets, const Tensor& predictions);

    /**
     * @brief Helper method to allocate the working spaces for loss.
     *
     * @param targets The expected output of the network.
     * @param predictions The actual output of the network.
     */
    void allocateWorkingSpacesLoss(const Tensor& targets, const Tensor& predictions);

public:
    std::string getShortName() const override;
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

    /**
     * @brief Space used when computing the loss/derivatives.
     */
private:
    Tensor workingSpace;

    /**
     * @brief Space containing only ones. Used when calculating the loss/derivatives.
     */
    Tensor onesLoss;

    /**
     * @brief Space containing only ones. Used when calculating the derivatives.
     */
    Tensor onesDerivatives;

    /**
     * @brief Tensor to store the sums of predictions.
     *
     * Used to normalize the predictions, such that their sum is 1.
     */
    Tensor accumulatedSumsLoss;

    /**
     * @brief Tensor to store the sums of predictions.
     *
     * Used to normalize the predictions, such that their sum is 1.
     */
    Tensor accumulatedSumsDerivatives;

    /**
     * @copybrief Loss::calculateLoss
     *
     * The loss is calculated for each data sample as
     * @f$\sum_{i=1}^n -y_i\ln(\frac{\hat{y}_i}{\sum_{j=1}^n \hat{y}_j}) @f$, where `n` is the number of classes.
     *
     * @copydetails Loss::calculateLoss
     */
public:
    float calculateLoss(const Tensor& targets, const Tensor& predictions) override;

    /**
     * @copybrief Loss::calculateDerivatives
     *
     * The derivative is calculated for each data sample as
     * @f$ -\frac{y_i}{\hat{y}_i} + \frac{1}{\sum_{j=1}^n \hat{y}_j} @f$.
     *
     * @copydetails Loss::calculateDerivatives
     */
    void calculateDerivatives(const Tensor& targets, const Tensor& predictions, Tensor& destination) override;

    /**
     * @brief Allocate the working spaces for calculating the loss.
     *
     * @param targets The expected output of the network.
     * @param predictions The actual output of the network.
     */
private:
    void allocateWorkingSpacesLoss(const Tensor& targets, const Tensor& predictions);

    /**
     * @brief Allocate the working spaces for calculating the derivatives.
     *
     * @param targets The expected output of the network.
     * @param predictions The actual output of the network.
     */
    void allocateWorkingSpacesDerivatives(const Tensor& targets, const Tensor& predictions);

public:
    std::string getShortName() const override;
};


#endif //NNLIB_LOSS_H
