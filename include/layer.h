//
// Created by Jan Warchocki on 03/03/2022.
//

#ifndef NNLIB_LAYER_H
#define NNLIB_LAYER_H

#include <string>
#include "matrix.h"
#include "activation.h"

#define DEFAULT_BATCH_SIZE 32

class Layer {
private:
    Matrix previousWeightsT;
    Matrix dataT;
    Vector ones;
    Matrix newDeltaT;

public:
    size_t outSize;
    size_t inSize;
    Activation* activation;

    Matrix weights;
    Vector biases;
    const Matrix* data;

    Matrix aMatrix;
    Matrix zMatrix;

    Matrix newDelta;
    Matrix derivatives;

    Matrix weightsGradients;
    Vector biasesGradients;

    Layer(size_t inSize, size_t outSize, Activation* activation);

    ~Layer();

    void forward(const Matrix& batch);

    void backward(const Matrix& delta, const Matrix& previousWeights, size_t batchSize = DEFAULT_BATCH_SIZE, bool isLastLayer = false);

    void applyGradients(size_t batchSize, DTYPE learningRate = 0.01);

private:
    void calculateDerivatives();

    void allocate(size_t batchSize);
};

#endif //NNLIB_LAYER_H
