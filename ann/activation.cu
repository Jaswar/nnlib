//
// Created by Jan Warchocki on 03/03/2022.
//

#include "activation.cuh"
#include "../gpu/allocation_gpu.cuh"

__global__
void ReLUDevice(DTYPE* vector, DTYPE* result, int n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    if (vector[index] <= 0) {
        result[index] = 0;
    } else {
        result[index] = vector[index];
    }
}

Vector ReLU(const Vector& v) {
    if (v.location == HOST) {
        DTYPE* newData = allocate1DArray(v.n);

        for (int i = 0; i < v.n; i++) {
            if (v[i] <= 0) {
                newData[i] = 0;
            } else {
                newData[i] = v[i];
            }
        }

        return Vector(newData, v.n);
    } else {
        DTYPE* result = allocate1DArrayDevice(v.n);

        ReLUDevice<<<1, v.n>>>(v.data, result, v.n);

        return Vector(result, v.n, DEVICE);
    }
}

__device__
DTYPE fSigmoidDevice(DTYPE x) {
    return 1 / (1 + expf(-x));
}

__global__
void sigmoidDevice(DTYPE* vector, DTYPE* result, int n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    result[index] = fSigmoidDevice(vector[index]);
}

DTYPE fSigmoid(DTYPE x) {
    return 1 / (1 + exp(-x));
}

Vector sigmoid(const Vector& v) {
    if (v.location == HOST) {
        DTYPE* newData = allocate1DArray(v.n);

        for (int i = 0; i < v.n; i++) {
            newData[i] = fSigmoid(v[i]);
        }

        return Vector(newData, v.n);
    } else {
        DTYPE* result = allocate1DArrayDevice(v.n);

        sigmoidDevice<<<1, v.n>>>(v.data, result, v.n);

        return Vector(result, v.n, DEVICE);
    }
}

Vector tanh(const Vector& v) {
    DTYPE* newData = allocate1DArray(v.n);

    for (int i = 0; i < v.n; i++) {
        newData[i] = tanh(v[i]);
    }

    return Vector(newData, v.n);
}

__global__
void linearDerivativeDevice(DTYPE* vector, DTYPE* result, int n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    result[index] = 1;
}

Vector linearDerivative(const Vector& input) {
    if (input.location == HOST) {
        DTYPE* newData = allocate1DArray(input.n, 1);
        return Vector(newData, input.n);
    } else {
        DTYPE* result = allocate1DArrayDevice(input.n);

        linearDerivativeDevice<<<1, input.n>>>(input.data, result, input.n);

        return Vector(result, input.n, DEVICE);
    }
}

__global__
void ReLUDerivativeDevice(DTYPE* vector, DTYPE* result, int n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    if (vector[index] <= 0) {
        result[index] = 0;
    } else {
        result[index] = 1;
    }
}

Vector ReLUDerivative(const Vector& input) {
    if (input.location == HOST) {
        DTYPE* newData = allocate1DArray(input.n);

        for (int i = 0; i < input.n; i++) {
            if (input[i] <= 0) {
                newData[i] = 0;
            } else {
                newData[i] = 1;
            }
        }

        return Vector(newData, input.n);
    } else {
        DTYPE* result = allocate1DArrayDevice(input.n);

        ReLUDerivativeDevice<<<1, input.n>>>(input.data, result, input.n);

        return Vector(result, input.n, DEVICE);
    }
}

__global__
void sigmoidDerivativeDevice(DTYPE* vector, DTYPE* result, int n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    result[index] = fSigmoidDevice(vector[index]) * (1 - fSigmoidDevice(vector[index]));
}

Vector sigmoidDerivative(const Vector& input) {
    if (input.location == HOST) {
        DTYPE* newData = allocate1DArray(input.n);

        for (int i = 0; i < input.n; i++) {
            newData[i] = fSigmoid(input[i]) * (1 - fSigmoid(input[i]));
        }

        return Vector(newData, input.n);
    } else {
        DTYPE* result = allocate1DArrayDevice(input.n);

        sigmoidDerivativeDevice<<<1, input.n>>>(input.data, result, input.n);

        return Vector(result, input.n, DEVICE);
    }
}

Vector tanhDerivative(const Vector& input) {
    DTYPE* newData = allocate1DArray(input.n);

    for (int i = 0; i < input.n; i++) {
        newData[i] = 1 - (tanh(input[i]) * tanh(input[i]));
    }

    return Vector(newData, input.n);
}
