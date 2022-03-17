//
// Created by Jan Warchocki on 03/03/2022.
//

#include "activation.cuh"
#include "../gpu/allocation_gpu.cuh"
#include "../exceptions/size_mismatch_exception.h"
#include "../exceptions/different_data_location_exception.h"

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

void ReLU(const Vector& v, Vector& result) {
    if (result.n != v.n) {
        throw SizeMismatchException();
    }
    if (result.location != v.location) {
        throw DifferentDataLocationException();
    }

    if (v.location == HOST) {
        for (int i = 0; i < v.n; i++) {
            if (v[i] <= 0) {
                result[i] = 0;
            } else {
                result[i] = v[i];
            }
        }
    } else {
        ReLUDevice<<<1, v.n>>>(v.data, result.data, v.n);
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

void sigmoid(const Vector& v, Vector& result) {
    if (result.n != v.n) {
        throw SizeMismatchException();
    }
    if (result.location != v.location) {
        throw DifferentDataLocationException();
    }

    if (v.location == HOST) {
        for (int i = 0; i < v.n; i++) {
            result[i] = fSigmoid(v[i]);
        }
    } else {
        sigmoidDevice<<<1, v.n>>>(v.data, result.data, v.n);
    }
}

__global__
void linearDevice(DTYPE* vector, DTYPE* result, int n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    result[index] = vector[index];
}

void linear(const Vector& v, Vector& result) {
    if (result.n != v.n) {
        throw SizeMismatchException();
    }
    if (result.location != v.location) {
        throw DifferentDataLocationException();
    }

    if (v.location == HOST) {
        for (int i = 0; i < v.n; i++) {
            result[i] = v[i];
        }
    } else {
        linearDevice<<<1, v.n>>>(v.data, result.data, v.n);
    }
}

__global__
void linearDerivativeDevice(DTYPE* vector, DTYPE* result, int n) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    result[index] = 1;
}

void linearDerivative(const Vector& input, Vector& result) {
    if (result.n != input.n) {
        throw SizeMismatchException();
    }
    if (result.location != input.location) {
        throw DifferentDataLocationException();
    }

    if (input.location == HOST) {
        for (int i = 0; i < input.n; i++) {
            result[i] = 1;
        }
    } else {
        linearDerivativeDevice<<<1, input.n>>>(input.data, result.data, input.n);
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

void ReLUDerivative(const Vector& input, Vector& result) {
    if (result.n != input.n) {
        throw SizeMismatchException();
    }
    if (result.location != input.location) {
        throw DifferentDataLocationException();
    }

    if (input.location == HOST) {
        for (int i = 0; i < input.n; i++) {
            if (input[i] <= 0) {
                result[i] = 0;
            } else {
                result[i] = 1;
            }
        }
    } else {
        ReLUDerivativeDevice<<<1, input.n>>>(input.data, result.data, input.n);
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

void sigmoidDerivative(const Vector& input, Vector& result) {
    if (result.n != input.n) {
        throw SizeMismatchException();
    }
    if (result.location != input.location) {
        throw DifferentDataLocationException();
    }

    if (input.location == HOST) {
        for (int i = 0; i < input.n; i++) {
            result[i] = fSigmoid(input[i]) * (1 - fSigmoid(input[i]));
        }
    } else {
        sigmoidDerivativeDevice<<<1, input.n>>>(input.data, result.data, input.n);
    }
}
