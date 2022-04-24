//
// Created by Jan Warchocki on 03/03/2022.
//

#include "activation.cuh"
#include "../gpu/allocation_gpu.cuh"
#include "../exceptions/size_mismatch_exception.h"
#include "../exceptions/different_data_location_exception.h"
#include "../gpu/assert.cuh"

__global__
void linearDevice(const DTYPE* vector, DTYPE* result, size_t n) {
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
        gpuCheckError( cudaGetLastError() )
        gpuCheckError( cudaDeviceSynchronize() )
    }
}

__global__
void linearDevice(const DTYPE* matrix, DTYPE* result, size_t n, size_t m) {
    auto row = blockIdx.x;
    auto column = threadIdx.x;

    if (row >= n || column >= m) {
        return;
    }

    result[row * m + column] = matrix[row * m + column];
}

void linear(const Matrix& m, Matrix& result) {
    if (m.n != result.n || m.m != result.m) {
        throw SizeMismatchException();
    }
    if (m.location != result.location) {
        throw DifferentDataLocationException();
    }

    if (m.location == HOST) {
        for (int row = 0; row < m.n; row++) {
            for (int i = 0; i < m.m; i++) {
                result(row, i) = m(row, i);
            }
        }
    } else {
        linearDevice<<<m.n, m.m>>>(m.data, result.data, m.n, m.m);
        gpuCheckError( cudaGetLastError() )
        gpuCheckError( cudaDeviceSynchronize() )
    }
}

__global__
void ReLUDevice(const DTYPE* vector, DTYPE* result, size_t n) {
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
        gpuCheckError( cudaGetLastError() )
        gpuCheckError( cudaDeviceSynchronize() )
    }
}

__global__
void ReLUDevice(const DTYPE* matrix, DTYPE* result, size_t n, size_t m) {
    auto row = blockIdx.x;
    auto column = threadIdx.x;

    if (row >= n || column >= m) {
        return;
    }

    if (matrix[row * m + column] <= 0) {
        result[row * m + column] = 0;
    } else {
        result[row * m + column] = matrix[row * m + column];
    }
}

void ReLU(const Matrix& m, Matrix& result) {
    if (m.n != result.n || m.m != result.m) {
        throw SizeMismatchException();
    }
    if (m.location != result.location) {
        throw DifferentDataLocationException();
    }

    if (m.location == HOST) {
        for (int row = 0; row < m.n; row++) {
            for (int i = 0; i < m.m; i++) {
                if (m(row, i) <= 0) {
                    result(row, i) = 0;
                } else {
                    result(row, i) = m(row, i);
                }
            }
        }
    } else {
        ReLUDevice<<<m.n, m.m>>>(m.data, result.data, m.n, m.m);
        gpuCheckError( cudaGetLastError() )
        gpuCheckError( cudaDeviceSynchronize() )
    }
}

__device__
DTYPE fSigmoidDevice(DTYPE x) {
    return 1 / (1 + expf(-x));
}

__global__
void sigmoidDevice(DTYPE* vector, DTYPE* result, size_t n) {
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
        gpuCheckError( cudaGetLastError() )
        gpuCheckError( cudaDeviceSynchronize() )
    }
}

__global__
void sigmoidDevice(DTYPE* matrix, DTYPE* result, size_t n, size_t m) {
    auto row = blockIdx.x;
    auto column = threadIdx.x;

    if (row >= n || column >= m) {
        return;
    }

    result[row * m + column] = fSigmoidDevice(matrix[row * m + column]);
}

void sigmoid(const Matrix& m, Matrix& result) {
    if (m.n != result.n || m.m != result.m) {
        throw SizeMismatchException();
    }
    if (result.location != m.location) {
        throw SizeMismatchException();
    }

    if (m.location == HOST) {
        for (int row = 0; row < m.n; row++) {
            for (int i = 0; i < m.m; i++) {
                result(row, i) = fSigmoid(m(row, i));
            }
        }
    } else {
        sigmoidDevice<<<m.n, m.m>>>(m.data, result.data, m.n, m.m);
        gpuCheckError( cudaGetLastError() )
        gpuCheckError( cudaDeviceSynchronize() )
    }
}

__global__
void linearDerivativeDevice(const DTYPE* vector, DTYPE* result, size_t n) {
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
        gpuCheckError( cudaGetLastError() )
        gpuCheckError( cudaDeviceSynchronize() )
    }
}

__global__
void linearDerivativeDevice(const DTYPE* matrix, DTYPE* result, size_t n, size_t m) {
    auto row = blockIdx.x;
    auto column = threadIdx.x;

    if (row >= n || column >= m) {
        return;
    }

    result[row * m + column] = 1;
}

void linearDerivative(const Matrix& input, Matrix& result) {
    if (input.n != result.n || input.m != result.m) {
        throw SizeMismatchException();
    }
    if (input.location != result.location) {
        throw DifferentDataLocationException();
    }

    if (input.location == HOST) {
        for (int row = 0; row < input.n; row++) {
            for (int i = 0; i < input.m; i++) {
                result(row, i) = 1;
            }
        }
    } else {
        linearDerivativeDevice<<<input.n, input.m>>>(input.data, result.data, input.n, input.m);
        gpuCheckError( cudaGetLastError() )
        gpuCheckError( cudaDeviceSynchronize() )
    }
}

__global__
void ReLUDerivativeDevice(const DTYPE* vector, DTYPE* result, size_t n) {
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
        gpuCheckError( cudaGetLastError() )
        gpuCheckError( cudaDeviceSynchronize() )
    }
}

__global__
void ReLUDerivativeDevice(const DTYPE* matrix, DTYPE* result, size_t n, size_t m) {
    auto row = blockIdx.x;
    auto column = threadIdx.x;

    if (row >= n || column >= m) {
        return;
    }

    if (matrix[row * m + column] <= 0) {
        result[row * m + column] = 0;
    } else {
        result[row * m + column] = 1;
    }
}

void ReLUDerivative(const Matrix& input, Matrix& result) {
    if (input.n != result.n || input.m != result.m) {
        throw SizeMismatchException();
    }
    if (result.location != input.location) {
        throw DifferentDataLocationException();
    }

    if (input.location == HOST) {
        for (int row = 0; row < input.n; row++) {
            for (int i = 0; i < input.m; i++) {
                if (input(row, i) <= 0) {
                    result(row, i) = 0;
                } else {
                    result(row, i) = 1;
                }
            }
        }
    } else {
        ReLUDerivativeDevice<<<input.n, input.m>>>(input.data, result.data, input.n, input.m);
        gpuCheckError( cudaGetLastError() )
        gpuCheckError( cudaDeviceSynchronize() )
    }
}

__global__
void sigmoidDerivativeDevice(DTYPE* vector, DTYPE* result, size_t n) {
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
        gpuCheckError( cudaGetLastError() )
        gpuCheckError( cudaDeviceSynchronize() )
    }
}

__global__
void sigmoidDerivativeDevice(DTYPE* matrix, DTYPE* result, size_t n, size_t m) {
    auto row = blockIdx.x;
    auto column = threadIdx.x;

    if (row >= n || column >= m) {
        return;
    }

    result[row * m + column] = fSigmoidDevice(matrix[row * m + column]) * (1 - fSigmoidDevice(matrix[row * m + column]));
}

void sigmoidDerivative(const Matrix& input, Matrix& result) {
    if (input.n != result.n || input.m != result.m) {
        throw SizeMismatchException();
    }
    if (input.location != result.location) {
        throw DifferentDataLocationException();
    }

    if (input.location == HOST) {
        for (int row = 0; row < input.n; row++) {
            for (int i = 0; i < input.m; i++) {
                result(row, i) = fSigmoid(input(row, i)) * (1 - fSigmoid(input(row, i)));
            }
        }
    } else {
        sigmoidDerivativeDevice<<<input.n, input.m>>>(input.data, result.data, input.n, input.m);
        gpuCheckError( cudaGetLastError() )
        gpuCheckError( cudaDeviceSynchronize() )
    }
}
