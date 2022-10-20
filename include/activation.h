/**
 * @file activation.h
 * @brief Header file declaring different activation functions.
 * @author Jan Warchocki
 * @date 03 March 2022
 */

#ifndef NNLIB_ACTIVATION_H
#define NNLIB_ACTIVATION_H

#include "tensor.h"
#include "verify.cuh"

/**
 * @brief Abstract class representing an evaluator of an activation function.
 *
 * Each activation function has two evaluators: one for on-host data and one for on-device data.
 * The correct evaluator is then decided based on the location of data passed into the activation function.
 *
 * Abstract away the evaluators into classes, so that the base Activation can choose the correct evaluator. In this
 * way code duplication is avoided in subclasses of Activation.
 */
class ActivationEvaluator {

    /**
     * @brief Method to apply the activation function on the input.
     *
     * This should apply the activation function to every element of the input.
     *
     * @param input The input.
     * @param result The input with the activation function applied to it.
     */
public:
    virtual void forward(const Tensor& input, Tensor& result) const = 0;

    /**
     * @brief Method to compute the derivatives given the output.
     *
     * @param output The output from a layer of the neural network.
     * @param result Where the computed derivatives should be written to.
     */
    virtual void computeDerivatives(const Tensor& output, Tensor& result) const = 0;

    /**
     * @brief The destructor.
     */
    virtual ~ActivationEvaluator() = default;
};

/**
 * @brief Abstract class representing an activation function.
 *
 * Other activation functions need to inherit from this class.
 *
 * As mentioned in ActivationEvaluator, the class contains evaluators for on-host and on-device data. The correct
 * evaluator is then chosen based on the location of data passed to the methods.
 */
class Activation {

    /**
     * @brief ActivationEvaluator for on-host data.
     */
protected:
    ActivationEvaluator* hostEvaluator;

    /**
     * @brief ActivationEvaluator for on-device data.
     */
    ActivationEvaluator* deviceEvaluator;

    /**
     * @brief Construct a new activation using both evaluators.
     *
     * @param hostEvaluator The evaluator for on-host data.
     * @param deviceEvaluator The evaluator for on-device data.
     */
public:
    explicit Activation(ActivationEvaluator* hostEvaluator, ActivationEvaluator* deviceEvaluator);

    /**
     * @brief Apply an activation function on the input.
     *
     * Calls ActivationEvaluator::forward() on the correct evaluator depending on if data is located on host or device.
     *
     * @param input The vector to apply activation function on.
     * @param result Where the applied activation function should be written to.
     *
     * @throws SizeMismatchException If sizes of the parameters don't match.
     * @throws DifferentDataLocationException If not all parameters are located in the same place.
     */
    void forward(const Tensor& input, Tensor& result) const;

    /**
     * @brief Compute the derivatives given the output of a layer of a neural network.
     *
     * Calls ActivationEvaluator::computeDerivatives() on the correct evaluator depending on
     * if data is located on host or device.
     *
     * @param output The values which derivatives will be computed.
     * @param result Where the derivatives should be written to.
     */
    void computeDerivatives(const Tensor& output, Tensor& result) const;

    /**
     * @brief Virtual destructor to make the class abstract.
     */
    virtual ~Activation() = 0;
};

/**
 * Linear activation declarations
 */

/**
 * @brief Represents a linear activation function.
 */
class LinearActivation : public Activation {
    /** @brief Construct a new LinearActivation. */
public:
    explicit LinearActivation();
};

/**
 * @brief On-host evaluator for the linear activation function.
 */
class LinearOnHostEvaluator : public ActivationEvaluator {

    /** @copydoc ActivationEvaluator::forward */
public:
    void forward(const Tensor& input, Tensor& result) const override;

    /** @copydoc ActivationEvaluator::computeDerivatives */
    void computeDerivatives(const Tensor& output, Tensor& result) const override;

    /** @brief The destructor. It is set to the default destructor. */
    ~LinearOnHostEvaluator() override;
};

/**
 * @brief On-device evaluator for the linear activation function.
 */
class LinearOnDeviceEvaluator : public ActivationEvaluator {

    /** @copydoc ActivationEvaluator::forward */
public:
    void forward(const Tensor& input, Tensor& result) const override;

    /** @copydoc ActivationEvaluator::computeDerivatives */
    void computeDerivatives(const Tensor& output, Tensor& result) const override;

    /** @brief The destructor. It is set to the default destructor. */
    ~LinearOnDeviceEvaluator() override;
};

/**
 * ReLU activation declarations
 */

/**
 * @brief Represents a ReLU activation function.
 */
class ReLUActivation : public Activation {

    /** @brief Construct a new ReLUActivation. */
public:
    explicit ReLUActivation();
};

/**
 * @brief On-host evaluator for the ReLU activation function.
 */
class ReLUOnHostEvaluator : public ActivationEvaluator {

    /** @copydoc ActivationEvaluator::forward */
public:
    void forward(const Tensor& input, Tensor& result) const override;

    /** @copydoc ActivationEvaluator::computeDerivatives */
    void computeDerivatives(const Tensor& output, Tensor& result) const override;

    /** @brief The destructor. It is set to the default destructor. */
    ~ReLUOnHostEvaluator() override;
};

/**
 * @brief On-device evaluator for the ReLU activation function.
 */
class ReLUOnDeviceEvaluator : public ActivationEvaluator {

    /** @copydoc ActivationEvaluator::forward */
public:
    void forward(const Tensor& input, Tensor& result) const override;

    /** @copydoc ActivationEvaluator::computeDerivatives */
    void computeDerivatives(const Tensor& output, Tensor& result) const override;

    /** @brief The destructor. It is set to the default destructor. */
    ~ReLUOnDeviceEvaluator() override;
};

/**
 * Sigmoid activation declarations
 */

/**
 * @brief Represents a sigmoid activation function.
 */
class SigmoidActivation : public Activation {

    /** @brief Construct a new SigmoidActivation. */
public:
    explicit SigmoidActivation();
};

/**
 * @brief On-host evaluator for the sigmoid activation function.
 */
class SigmoidOnHostEvaluator : public ActivationEvaluator {

    /** @copydoc ActivationEvaluator::forward */
public:
    void forward(const Tensor& input, Tensor& result) const override;

    /** @copydoc ActivationEvaluator::computeDerivatives */
    void computeDerivatives(const Tensor& output, Tensor& result) const override;

    /** @brief The destructor. It is set to the default destructor. */
    ~SigmoidOnHostEvaluator() override;
};

/**
 * @brief On-device evaluator for the sigmoid activation function.
 */
class SigmoidOnDeviceEvaluator : public ActivationEvaluator {

    /** @copydoc ActivationEvaluator::forward */
public:
    void forward(const Tensor& input, Tensor& result) const override;

    /** @copydoc ActivationEvaluator::computeDerivatives */
    void computeDerivatives(const Tensor& output, Tensor& result) const override;

    /** @brief The destructor. It is set to the default destructor. */
    ~SigmoidOnDeviceEvaluator() override;
};

#endif //NNLIB_ACTIVATION_H
