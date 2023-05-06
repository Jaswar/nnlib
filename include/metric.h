/**
 * @file metric.h
 * @brief Header file declaring different metrics.
 * @author Jan Warchocki
 * @date 13 February 2023
 */

#ifndef NNLIB_METRIC_H
#define NNLIB_METRIC_H

#include "tensor.h"

/**
 * @brief An abstract class to represent metrics.
 *
 * The metric function is called after every batch is processed and reset at the end of each epoch. The metric
 * keeps track of the current metric in an epoch by maintaining the number of datapoints processed and the total metric.
 */
class Metric {

    /**
     * @brief The number of samples processed so far.
     */
protected:
    size_t numSamples;

    /**
     * @brief The current total value of the metric.
     */
    float currentTotalMetric;

    /**
     * @brief Constructor for the Metric class.
     *
     * All it does is initialize `numSamples` and `currentTotalMetric` to 0.
     */
public:
    Metric();

    /**
     * @brief Reset the metric, i.e.: set `numSamples` and `currentTotalMetric` to 0.
     */
    void reset();

    /**
     * @brief Calcualate the current value of the metric given the new batches of targets and predictions.
     *
     * @param targets The desired outputs of the network.
     * @param predictions The actual outputs of the network.
     * @return The value of the metric.
     */
    virtual float calculateMetric(const Tensor& targets, const Tensor& predictions) = 0;

    /**
     * @brief Short string identifier of the metric.
     *
     * Used when printing the value of the metric to the terminal.
     *
     * @return A string identifier of the metric.
     */
    virtual std::string getShortName() const = 0;
};

/**
 * @brief The implementation of categorical accuracy.
 *
 * This metric requires the targets to consist of only 0s and 1s, with 1s corresponding to the correct class.
 * The rows are the samples, the columns are the classes. The predictions can consist of any real value, the largest
 * value is assumed to be the predicted class.
 */
class CategoricalAccuracy : public Metric {

    /**
     * @brief Constructor of CategoricalAccuracy.
     */
public:
    CategoricalAccuracy();

    /**
     * @copydoc Metric::calculateMetric()
     */
    float calculateMetric(const Tensor& targets, const Tensor& predictions) override;

    /**
     * @copydoc Metric::getShortName()
     */
    std::string getShortName() const override;
};

/**
 * @brief The implementation of binary accuracy.
 *
 * This metric assumes the targets to be of shape Nx1 with N being the batch size. The targets can only consist of
 * 0s and 1s corresponding to the two classes. The predictions can be any real value from the range [0, 1].
 * Predictions larger than 0.5 are assigned to class 1 while predictions smaller than 0.5 are assigned to class 0.
 */
class BinaryAccuracy : public Metric {

    /**
     * @brief Constructor of BinaryAccuracy.
     */
public:
    BinaryAccuracy();

    /**
     * @copydoc Metric::calculateMetric()
     */
    float calculateMetric(const Tensor& targets, const Tensor& predictions) override;

    /**
     * @copydoc Metric::getShortName()
     */
    std::string getShortName() const override;
};

#endif //NNLIB_METRIC_H
