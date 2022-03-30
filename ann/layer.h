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
    const Matrix* data;

    Matrix aMatrix;
    Matrix zMatrix;

    Matrix newDelta;
    Matrix derivatives;

    Matrix weightsGradients;
    Vector biasesGradients;

    Layer(int inSize, int outSize, const std::string& activation = "linear");

    ~Layer();

    void forward(const Matrix& batch);

    void backward(const Matrix& delta, const Matrix& previousWeights,
                                       bool isLastLayer = false, DTYPE learningRate = 0.01);

    void applyGradients();

    void calculateDerivatives();
};

#endif //NNLIB_LAYER_H
