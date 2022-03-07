//
// Created by Jan Warchocki on 03/03/2022.
//

#ifndef NNLIB_NETWORK_H
#define NNLIB_NETWORK_H

#include <vector>
#include "layer.h"

#define NO_SEED -1

class Network {
    std::vector<Layer> layers;
    long long seed;
    int previousSize;

public:
    explicit Network(int inputSize, long long seed = NO_SEED);

    void add(int numNeurons, const std::string& activation = "linear");

    Vector forward(const Vector& input);

    void backward(const Vector& predicted, const Vector& target, DTYPE learningRate = 0.01);

    void train(const Matrix& X, const Matrix& y, int epochs, DTYPE learningRate = 0.01);
};


#endif //NNLIB_NETWORK_H
