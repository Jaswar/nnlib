/**
 * @file layer.h
 * @brief Header file declaring the Layer class.
 * @author Jan Warchocki
 * @date 03 March 2022
 */

#ifndef NNLIB_LAYER_H
#define NNLIB_LAYER_H

#include "activation.h"
#include "matrix.h"
#include <string>

/**
 * @brief The default batch size if no batch size is specified.
 *
 * This macro is also used to pre-allocate space when a layer is first created. In this way all required
 * matrices/vectors are initialized and may only need reshaping later when starting training.
 */
#define DEFAULT_BATCH_SIZE 32

/**
 * @brief Represents a single layer of a neural network.
 */
class Layer {

    /**
     * @brief %Matrix storing the transpose of the weights of the previous layer.
     *
     * Helper variable used during backpropagation.
     */
private:
    Matrix previousWeightsT;

    /**
     * @brief %Matrix storing the transpose of the data passed in the forward propagation step.
     *
     * Helper variable used during backpropagation.
     */
    Matrix dataT;

    /**
     * @brief Store a vector of ones.
     *
     * Required for the backpropagation algorithm. It is used to sum Layer::newDeltaT along first axis into
     * bias gradients.
     */
    Vector ones;

    /**
     * @brief Transpose of Layer::newDelta.
     *
     * Helper variable used during backpropagation.
     */
    Matrix newDeltaT;

    /**
     * @brief The location of the layer.
     *
     * Specifies the location of all the data used by the layer. See DataLocation for more info.
     */
public:
    DataLocation location;

    /**
     * @brief The output size of the layer.
     *
     * Equal to the number of neurons in the layer.
     */
    size_t outSize;

    /**
     * @brief The input size to the layer.
     *
     * Equal to the number of neurons in the previous layer, or the size of the input in the case of the input layer.
     */
    size_t inSize;

    /**
     * @brief The activation function.
     *
     * Pointer to the activation function object. Can be LinearActivation, ReLUActivation or SigmoidActivation.
     */
    Activation* activation;

    /**
     * @brief The weights of the layer.
     */
    Matrix weights;

    /**
     * @brief The biases of the layer.
     */
    Vector biases;

    /**
     * @brief %Matrix storing data passed to the layer.
     *
     * Stores a pointer reference to the batch that was most recently forward-propagated through the layer.
     */
    const Matrix* data;

    /**
     * @brief The output of the layer before applying the activation function.
     */
    Matrix aMatrix;

    /**
     * @brief The output of the layer.
     */
    Matrix zMatrix;

    /**
     * @brief Delta that should be passed to the previous layer in the backpropagation step.
     */
    Matrix newDelta;

    /**
     * @brief The derivatives of the output.
     *
     * The derivatives are computed by the activation function and stored in this variable.
     */
    Matrix derivatives;

    /**
     * @brief The weights gradients computed by the backpropagation algorithm.
     */
    Matrix weightsGradients;

    /**
     * @brief The biases gradients computed by the backpropagation algorithm.
     */
    Vector biasesGradients;

    /**
     * @brief Construct a new layer.
     *
     * Also allocates space that will be used during computation. This allows for in-place computation, which
     * avoids allocating/freeing memory during training.
     *
     * @param inSize The input size to the layer.
     * @param outSize The output size of the layer (equal to the number of neurons).
     * @param activation The activation function that should be used.
     * @param location The location of the layer. See Layer::location.
     */
    Layer(size_t inSize, size_t outSize, Activation* activation, DataLocation location);

    /**
     * @brief The destructor of the layer object.
     */
    ~Layer();

    /**
     * @brief Forward one batch of data through the layer.
     *
     * This includes allocating space that could not be allocated in the constructor as it depends on the batch size.
     * The additional data will only be allocated if batch size changes. This means, if all batches are of the same
     * size, the data will not be allocated again. This is performed in the Layer::allocate() method.
     *
     * @param batch The batch that should be propagated.
     */
    void forward(const Matrix& batch);

    /**
     * @brief Backward-propagate one batch of data through the network.
     *
     * Takes a boolean to specify if this layer is the output layer in the network. If it is, a slightly
     * different algorithm must be used to compute the gradients.
     *
     * This method only computes the gradients, it does not apply them. The gradients can only be applied once
     * they have been calculated for all the layers. Otherwise, the passed @p previousWeights would change before
     * the gradients have been computed. The gradients are applied in the Layer::applyGradients() method.
     *
     * Uses algorithm adapted from http://neuralnetworksanddeeplearning.com/chap2.html.
     *
     * @param delta @p newDelta passed from the previous layer (next in the model's architecture).
     * @param previousWeights The weights of the previous layer (next in the model's architecture).
     * @param batchSize The size of the batch.
     * @param isLastLayer Boolean to specify if this layer is the last one (the output layer).
     */
    void backward(const Matrix& delta, const Matrix& previousWeights, size_t batchSize = DEFAULT_BATCH_SIZE,
                  bool isLastLayer = false);

    /**
     * @brief Apply the computed gradients.
     *
     * The method should be called only when all the gradients have been computed for all the layers in the network.
     *
     * @param batchSize The size of the batch.
     * @param learningRate The learning rate of the model.
     */
    void applyGradients(size_t batchSize, DTYPE learningRate = 0.01);

    /**
     * @brief Calculate the derivatives of the output.
     *
     * This calls Activation::computeDerivatives() on the Layer::activation object.
     */
private:
    void calculateDerivatives();

    /**
     * @brief Allocate data required for computation.
     *
     * Step called during forward propagation. If some matrices are in incorrect shape, this method will reallocate
     * their memory to match the correct shape. This method calls all <em>allocate*</em> methods.
     *
     * @param batchSize The size of the batch.
     */
    void allocate(size_t batchSize);

    /**
     * @brief Allocate Layer::ones.
     *
     * @param batchSize The size of the batch.
     */
    void allocateOnes(size_t batchSize);

    /**
     * @brief Allocate Layer::dataT.
     *
     * @param batchSize The size of the batch.
     */
    void allocateDataT(size_t batchSize);

    /**
     * @brief Allocate Layer::aMatrix.
     *
     * @param batchSize The size of the batch.
     */
    void allocateAMatrix(size_t batchSize);

    /**
     * @brief Allocate Layer::zMatrix.
     *
     * @param batchSize The size of the batch.
     */
    void allocateZMatrix(size_t batchSize);

    /**
     * @brief Allocate Layer::newDelta.
     *
     * @param batchSize The size of the batch.
     */
    void allocateNewDelta(size_t batchSize);

    /**
     * @brief Allocate Layer::newDeltaT.
     *
     * @param batchSize The size of the batch.
     */
    void allocateNewDeltaT(size_t batchSize);

    /**
     * @brief Allocate Layer::derivatives.
     *
     * @param batchSize The size of the batch.
     */
    void allocateDerivatives(size_t batchSize);
};

#endif //NNLIB_LAYER_H
