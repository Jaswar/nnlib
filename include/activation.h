//
// Created by Jan Warchocki on 03/03/2022.
//

#ifndef NNLIB_ACTIVATION_H
#define NNLIB_ACTIVATION_H

#include "vector.h"
#include "verify.cuh"
#include "matrix.h"

class ActivationEvaluator {
public:
    virtual void forward(const Vector& input, Vector& result) const = 0;
    virtual void forward(const Matrix& input, Matrix& result) const = 0;

    virtual void computeDerivatives(const Vector& output, Vector& result) const = 0;
    virtual void computeDerivatives(const Matrix& output, Matrix& result) const = 0;

    virtual ~ActivationEvaluator() = default;
};

class Activation {
protected:
    ActivationEvaluator* evaluator;
    DataLocation location;

public:
    explicit Activation(DataLocation location);

    void forward(const Vector& input, Vector& result) const;
    void forward(const Matrix& input, Matrix& result) const;

    void computeDerivatives(const Vector& output, Vector& result) const;
    void computeDerivatives(const Matrix& output, Matrix& result) const;

    virtual ~Activation() = 0;
};

/**
 * Linear activation declarations
 */

class LinearActivation : public Activation {
public:
    explicit LinearActivation(DataLocation location);
};

class LinearOnHostEvaluator : public ActivationEvaluator {
public:
    void forward(const Vector& input, Vector& result) const override;
    void forward(const Matrix& input, Matrix& result) const override;
    void computeDerivatives(const Vector& output, Vector& result) const override;
    void computeDerivatives(const Matrix& output, Matrix& result) const override;

    ~LinearOnHostEvaluator() override;
};

class LinearOnDeviceEvaluator : public ActivationEvaluator {
public:
    void forward(const Vector& input, Vector& result) const override;
    void forward(const Matrix& input, Matrix& result) const override;
    void computeDerivatives(const Vector& output, Vector& result) const override;
    void computeDerivatives(const Matrix& output, Matrix& result) const override;

    ~LinearOnDeviceEvaluator() override;
};

/**
 * ReLU activation declarations
 */

class ReLUActivation : public Activation {
public:
    explicit ReLUActivation(DataLocation location);
};

class ReLUOnHostEvaluator : public ActivationEvaluator {
public:
    void forward(const Vector& input, Vector& result) const override;
    void forward(const Matrix& input, Matrix& result) const override;
    void computeDerivatives(const Vector& output, Vector& result) const override;
    void computeDerivatives(const Matrix& output, Matrix& result) const override;

    ~ReLUOnHostEvaluator() override;
};

class ReLUOnDeviceEvaluator : public ActivationEvaluator {
public:
    void forward(const Vector& input, Vector& result) const override;
    void forward(const Matrix& input, Matrix& result) const override;
    void computeDerivatives(const Vector& output, Vector& result) const override;
    void computeDerivatives(const Matrix& output, Matrix& result) const override;

    ~ReLUOnDeviceEvaluator() override;
};

/**
 * Sigmoid activation declarations
 */

class SigmoidActivation : public Activation {
public:
    explicit SigmoidActivation(DataLocation location);
};

class SigmoidOnHostEvaluator : public ActivationEvaluator {
public:
    void forward(const Vector& input, Vector& result) const override;
    void forward(const Matrix& input, Matrix& result) const override;
    void computeDerivatives(const Vector& output, Vector& result) const override;
    void computeDerivatives(const Matrix& output, Matrix& result) const override;

    ~SigmoidOnHostEvaluator() override;
};

class SigmoidOnDeviceEvaluator : public ActivationEvaluator {
public:
    void forward(const Vector& input, Vector& result) const override;
    void forward(const Matrix& input, Matrix& result) const override;
    void computeDerivatives(const Vector& output, Vector& result) const override;
    void computeDerivatives(const Matrix& output, Matrix& result) const override;

    ~SigmoidOnDeviceEvaluator() override;
};

#endif //NNLIB_ACTIVATION_H
