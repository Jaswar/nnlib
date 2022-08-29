/**
 * @file tensor.h
 * @brief Header file declaring the Tensor class to represent multidimensional arrays.
 * @author Jan Warchocki
 * @date 26 August 2022
 */

#ifndef NNLIB_TENSOR_H
#define NNLIB_TENSOR_H

#include <cstdlib>
#include <vector>
#include "vector.h"

class Tensor {
public:
    std::vector<size_t> shape;
    size_t size;

    DataLocation location;
    union {
        float* host;
        float* device;
    };

    Tensor();

    Tensor(const Tensor& other);

    template<typename... Args>
    explicit Tensor(Args... args) : shape({static_cast<size_t>(args)...}), location(HOST), size(0), device(nullptr) {
        computeSize();
        host = allocate1DArray(size, 0);
    }

    Tensor& operator=(const Tensor& other);

    void moveToDevice();
    void moveToHost();

    ~Tensor();

private:
    void computeSize();
};

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor);

void add(const Tensor& a, const Tensor& b, Tensor& destination);
void subtract(const Tensor& a, const Tensor& b, Tensor& destination);
void hadamard(const Tensor& a, const Tensor& b, Tensor& destination);

void multiply(const Tensor& tensor, float constant, Tensor& destination);

void multiply(const Tensor& a, const Tensor& b, Tensor& destination);

void transpose(const Tensor& matrix, Tensor& destination);

#endif //NNLIB_TENSOR_H
