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
    Vector data;

    Vector aVector;

    Vector delta;

    Layer(int inSize, int outSize, const std::string& activation = "linear");

    ~Layer();

    Vector forward(const Vector& input);

    std::pair<Vector, Matrix> backward(const Vector& delta, const Matrix& previousWeights,
                                       bool isLastLayer = false, DTYPE learningRate = 0.01);

    Vector calculateDerivatives() const;

    void build(int previousSize);
};

#endif //NNLIB_LAYER_H
