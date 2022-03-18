//
// Created by Jan Warchocki on 03/03/2022.
//

#ifndef NNLIB_LAYER_H
#define NNLIB_LAYER_H

#include <string>
#include "../math/matrix.h"

class Layer {
public:
    int outSize;
    int inSize;
    std::string activation;

    Matrix weights;
    Vector biases;
    const Vector* data;

    Vector aVector;
    Vector zVector;

    Vector newDelta;
    Vector derivatives;

    Layer(int inSize, int outSize, const std::string& activation = "linear");

    ~Layer();

    void forward(const Vector& input);

    void backward(const Vector& delta, const Matrix& previousWeights,
                                       bool isLastLayer = false, DTYPE learningRate = 0.01);

    void calculateDerivatives();
};

#endif //NNLIB_LAYER_H
