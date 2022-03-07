//
// Created by Jan Warchocki on 03/03/2022.
//

#include "activation.h"

Vector ReLU(const Vector& v) {
    DTYPE* newData = allocate1DArray(v.n);

    for (int i = 0; i < v.n; i++) {
        if (v[i] <= 0) {
            newData[i] = 0;
        } else {
            newData[i] = v[i];
        }
    }

    return Vector(newData, v.n);
}

DTYPE sigmoid(DTYPE x) {
    return 1 / (1 + exp(-x));
}

Vector sigmoid(const Vector& v) {
    DTYPE* newData = allocate1DArray(v.n);

    for (int i = 0; i < v.n; i++) {
        newData[i] = sigmoid(v[i]);
    }

    return Vector(newData, v.n);
}

Vector tanh(const Vector& v) {
    DTYPE* newData = allocate1DArray(v.n);

    for (int i = 0; i < v.n; i++) {
        newData[i] = tanh(v[i]);
    }

    return Vector(newData, v.n);
}

Vector ReLUDerivative(const Vector& input) {
    DTYPE* newData = allocate1DArray(input.n);

    for (int i = 0; i < input.n; i++) {
        if (input[i] <= 0) {
            newData[i] = 0;
        } else {
            newData[i] = 1;
        }
    }

    return Vector(newData, input.n);
}

Vector sigmoidDerivative(const Vector& input) {
    DTYPE* newData = allocate1DArray(input.n);

    for (int i = 0; i < input.n; i++) {
        newData[i] = sigmoid(input[i]) * (1 - sigmoid(input[i]));
    }

    return Vector(newData, input.n);
}

Vector tanhDerivative(const Vector& input) {
    DTYPE* newData = allocate1DArray(input.n);

    for (int i = 0; i < input.n; i++) {
        newData[i] = 1 - (tanh(input[i]) * tanh(input[i]));
    }

    return Vector(newData, input.n);
}
