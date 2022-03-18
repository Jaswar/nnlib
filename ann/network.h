//
// Created by Jan Warchocki on 03/03/2022.
//

#ifndef NNLIB_NETWORK_H
#define NNLIB_NETWORK_H

#include <vector>
#include "layer.h"

#define NO_SEED -1

class Network {
    // Pre-allocated space for loss
    Vector loss;

    std::vector<Layer> layers;
    long long seed;
    int previousSize;

public:
    explicit Network(int inputSize, long long seed = NO_SEED);

    void add(int numNeurons, const std::string& activation = "linear");

    void forward(const Vector& input, Vector& output);

    void backward(const Vector& predicted, const Vector& target, DTYPE learningRate = 0.01);

    void train(const Matrix& X, const Matrix& y, int epochs, DTYPE learningRate = 0.01);
};


#endif //NNLIB_NETWORK_H
