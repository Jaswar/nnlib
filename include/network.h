//
// Created by Jan Warchocki on 03/03/2022.
//

#ifndef NNLIB_NETWORK_H
#define NNLIB_NETWORK_H

#include "layer.h"
#include <vector>

#define NO_SEED (-1)

class Network {
    DataLocation location;

    // Pre-allocated space for loss
    Matrix loss;

    std::vector<Layer> layers;
    long long seed;
    size_t previousSize;

public:
    explicit Network(size_t inputSize, bool useGPU = true, long long seed = NO_SEED);

    void add(size_t numNeurons, const std::string& activation = "linear");

    Matrix* forward(const Matrix& batch);

    void backward(const Matrix& predicted, const Matrix& target, DTYPE learningRate = 0.01);

    //NOLINTNEXTLINE(readability-identifier-naming)
    void train(const Matrix& X, const Matrix& y, int epochs, size_t batchSize = DEFAULT_BATCH_SIZE,
               DTYPE learningRate = 0.01);

private:
    //NOLINTNEXTLINE(readability-identifier-naming)
    static void computeAccuracy(const Matrix& X, const Matrix& yHost, const Matrix& predictions);
};


#endif //NNLIB_NETWORK_H
