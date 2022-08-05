/**
 * @file network.h
 * @brief Header file declaring the Network class.
 * @author Jan Warchocki
 * @date 03 March 2022
 */

#ifndef NNLIB_NETWORK_H
#define NNLIB_NETWORK_H

#include "layer.h"
#include <vector>

/**
 * @brief Integer to mean that no seed was specified for the network.
 */
#define NO_SEED (-1)

/**
 * @brief Represents a neural network.
 */
class Network {
    /**
     * @brief The location of the network.
     *
     * Specifies the location of all the data used by the network. See DataLocation for more info.
     */
    DataLocation location;

    /**
     * @brief Pre-allocated space for loss.
     *
     * Might require resizing/reallocating if @p batchSize != #DEFAULT_BATCH_SIZE during training. In that case,
     * the reshaping will still only happen once. The data is pre-allocated to avoid unnecessary allocation
     * during runtime.
     */
    Matrix loss;

    /**
     * @brief List of network layers.
     */
    std::vector<Layer> layers;

    /**
     * @brief Seed used for random initialization.
     */
    long long seed;

    /**
     * @brief Keeps track of the size of the previous layer.
     *
     * Layers need to know the size of the input passed to them. This is achieved using this variable, that
     * keeps track of the size of the previous layer (or size of the input if in 1st layer). In this way, layers
     * can pre-allocate required space during initialization.
     */
    size_t previousSize;

    /**
     * @brief Construct a new network.
     *
     * The constructed network can use GPU acceleration if a GPU and CUDA are available, and the @p useGPU parameter
     * is set to true.
     *
     * @param inputSize The number of inputs to the neural network.
     * @param useGPU Boolean to specify whether the network should use GPU acceleration.
     * @param seed Seed that should be used for random initialization of the network.
     */
public:
    explicit Network(size_t inputSize, bool useGPU = true, long long seed = NO_SEED);

    /**
     * @brief Add a new layer to the network.
     *
     * Three activation functions can be used: Linear, ReLU and Sigmoid. The activation can be selected
     * by specifying the activation parameter, using one of the strings: "linear", "relu" or "sigmoid".
     * If any other string is specified, Linear activation will be used.
     *
     * @param numNeurons The number of neurons the new layer should contain.
     * @param activation The activation function to use. Can be "linear", "relu" or "sigmoid".
     */
    void add(size_t numNeurons, const std::string& activation = "linear");

    /**
     * @brief Forward-propagate a batch through the network.
     *
     * The samples should be aligned along the first axis.
     *
     * @param batch The batch to propagate.
     * @return The pointer to the output of the network. This returns Layer::aMatrix of the last layer.
     */
    Matrix* forward(const Matrix& batch);

    /**
     * @brief Backward-propagate a batch through the network.
     *
     * Squared error loss is used as the loss metric. The network first calculates the gradients on
     * all layers and only then applies them. This is because layers require weights from following
     * layers to compute the correct gradients.
     *
     * @param predicted The predictions of the network as retrieved from Network::forward.
     * @param target The targets for that batch of data.
     * @param learningRate The learning rate of the model.
     */
    void backward(const Matrix& predicted, const Matrix& target, DTYPE learningRate = 0.01);

    /**
     * @brief Train the network.
     *
     * Both X and y should have the data samples aligned on the first axis. Each row in X should be aligned
     * with the corresponding row in y.
     *
     * The only supported metric for now is accuracy, which is calculated by default by the method.
     * Furthermore, the method displays the progress of each epoch, including time spent and accuracy.
     *
     * @param X The data to train the network on.
     * @param y The targets of the network.
     * @param epochs The number of epochs to train the network for.
     * @param batchSize The size of the batch.
     * @param learningRate The learning rate of the algorithm.
     */
    //NOLINTNEXTLINE(readability-identifier-naming)
    void train(const Matrix& X, const Matrix& y, int epochs, size_t batchSize = DEFAULT_BATCH_SIZE,
               DTYPE learningRate = 0.01);

private:
    /**
     * @brief Trains the model on a single epoch.
     *
     * Helper method used in Network::train(). The method computes accuracy, which is why yHost is passed as an argument.
     * In this way, when the network is running on GPU, the targets will not have to be copied to host memory when
     * computing accuracy.
     *
     * @param batches The list of batches to process. These have been split in Network::train() method.
     * @param targets The list of targets to process. These have been split in Network::train() method.
     * @param yHost Matrix that stores the whole y array on host.
     * @param learningRate The learning rate used during training.
     */
    void processEpoch(std::vector<Matrix>& batches, std::vector<Matrix>& targets, Matrix& yHost, DTYPE learningRate);
};


#endif //NNLIB_NETWORK_H
