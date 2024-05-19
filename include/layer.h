/**
 * @file layer.h
 * @brief Header file declaring the Layer class.
 * @author Jan Warchocki
 * @date 03 March 2022
 */

#ifndef NNLIB_LAYER_H
#define NNLIB_LAYER_H

#include "tensor.h"
#include <string>

/**
 * @brief Represents a single layer of a neural network.
 */
class Layer {
    /**
     * @brief The location of the layer.
     *
     * Specifies the location of all the data used by the layer. See #DataLocation for more info.
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
    std::string activation;

    /**
     * @brief The weights of the layer. Stored as a matrix.
     */
    sfTensor weights;

    /**
     * @brief The biases of the layer. Stored as a vector.
     */
    sfTensor biases;

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
    Layer(size_t inSize, size_t outSize, std::string activation, DataLocation location);

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
    // You might want to ignore the return value of forward, so don't use [[nodiscard]]
    // NOLINTNEXTLINE(modernize-use-nodiscard)
    sfTensor forward(const sfTensor& batch) const;

    /**
     * @brief Apply the computed gradients.
     *
     * The method should be called only when all the gradients have been computed for all the layers in the network.
     *
     * @param batchSize The size of the batch.
     * @param learningRate The learning rate of the model.
     */
    void applyGradients(size_t batchSize, float learningRate = 0.01);
};

#endif //NNLIB_LAYER_H
