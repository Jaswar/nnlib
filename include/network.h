/**
 * @file network.h
 * @brief Header file declaring the Network class.
 * @author Jan Warchocki
 * @date 03 March 2022
 */

#ifndef NNLIB_NETWORK_H
#define NNLIB_NETWORK_H

#include "layer.h"
#include "loss.h"
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
     * Specifies the location of all the data used by the network. See #DataLocation for more info.
     */
    DataLocation location;

    /**
     * @brief Pre-allocated space for loss.
     *
     * Might require resizing/reallocating if @p batchSize != #DEFAULT_BATCH_SIZE during training. In that case,
     * the reshaping will still only happen once. The data is pre-allocated to avoid unnecessary allocation
     * during runtime.
     */
    Tensor lossData;

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
    Tensor* forward(const Tensor& batch);

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
     * @param loss The loss function to use.
     */
    void backward(const Tensor& predicted, const Tensor& target, float learningRate, Loss* loss);

    /**
     * @brief Train the network.
     *
     * Both X and y should have the data samples aligned on the first axis. Each row in X should be aligned
     * with the corresponding row in y.
     *
     * @param X The data to train the network on.
     * @param y The targets of the network.
     * @param epochs The number of epochs to train the network for.
     * @param batchSize The size of the batch.
     * @param learningRate The learning rate of the algorithm.
     * @param loss The loss function to use.
     * @param metrics The list of metrics to compute aside from the loss function.
     */
    //NOLINTNEXTLINE(readability-identifier-naming)
    void train(Tensor& X, Tensor& y, int epochs, size_t batchSize, float learningRate, Loss* loss,
               std::vector<Metric*>& metrics);

private:
    /**
     * @brief Trains the model on a single epoch.
     *
     * Helper method used in Network::train(). The method makes use of `targetsOnHost`, which are the target batches
     * stored on host. This is for performance reasons as some of the metrics require the input matrices to be located
     * on host.
     *
     * @param batches The list of batches to process. These have been split in Network::train() method.
     * @param targets The list of targets to process. These have been split in Network::train() method.
     * @param targetsOnHost The list of targets to processed but stored on host.
     * @param learningRate The learning rate used during training.
     * @param loss The loss function to use.
     * @param metrics The list of metrics to compute aside from the loss function.
     */
    void processEpoch(std::vector<Tensor>& batches, std::vector<Tensor>& targets, std::vector<Tensor>& targetsOnHost,
                      float learningRate, Loss* loss, std::vector<Metric*>& metrics);
};


#endif //NNLIB_NETWORK_H
