/**
 * @file metric.h
 * @brief 
 * @author Jan Warchocki
 * @date 13 February 2023
 */

#ifndef NNLIB_METRIC_H
#define NNLIB_METRIC_H

#include "tensor.h"

class Metric {
protected:
    size_t numSamples;
    float currentTotalMetric;

public:
    Metric();
    void reset();

    virtual float calculateMetric(const Tensor& targets, const Tensor& predictions) = 0;

    virtual std::string getShortName() const = 0;
};

class CategoricalAccuracy : public Metric {
public:
    CategoricalAccuracy();

    float calculateMetric(const Tensor& targets, const Tensor& predictions) override;

    std::string getShortName() const override;
};

class BinaryAccuracy : public Metric {
public:
    BinaryAccuracy();

    float calculateMetric(const Tensor& targets, const Tensor& predictions) override;

    std::string getShortName() const override;
};

#endif //NNLIB_METRIC_H
