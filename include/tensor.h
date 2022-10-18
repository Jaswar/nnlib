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
#include "session.h"
#include "allocation.h"
#include "../src/exceptions/size_mismatch_exception.h"
#include <iostream>

enum DataLocation {
    HOST, DEVICE
};

class Tensor {
public:
    std::vector<size_t> shape;
    size_t size;

    DataLocation location;
    union {
        float* host;
        float* device;
    };

    Session session;

    Tensor();

    Tensor(const Tensor& other);

    template<typename... Args>
    explicit Tensor(Args... args) : shape({static_cast<size_t>(args)...}), location(HOST), size(0), device(nullptr) {
        computeSize();
        host = allocate1DArray(size, 0);
    }

    Tensor& operator=(const Tensor& other);

    void move(DataLocation target);

    static Tensor construct1d(const std::vector<float>& data);
    static Tensor construct2d(const std::vector<std::vector<float>>& data);

    template<typename... Args>
    float& operator()(Args... args) {
        std::vector<size_t> index = std::vector<size_t>({static_cast<size_t>(args)...});
        // Make sure the indexes are within acceptable range
        if (index.size() != shape.size()) {
            throw SizeMismatchException();
        }
        for (size_t i = 0; i < index.size(); i++) {
            if (index[i] >= shape[i]) {
                throw SizeMismatchException();
            }
        }
        // Recursively figure out the index in the flattened array (the effective index)
        size_t effectiveIndex = findEffectiveAddress(index, shape.size() - 1);
        return host[effectiveIndex];
    }

    ~Tensor();

private:
    void computeSize();
    size_t findEffectiveAddress(std::vector<size_t> index, size_t depth) const;
};

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor);

void fill(float value, Tensor& destination);

void add(const Tensor& a, const Tensor& b, Tensor& destination);
void subtract(const Tensor& a, const Tensor& b, Tensor& destination);
void hadamard(const Tensor& a, const Tensor& b, Tensor& destination);

void multiply(const Tensor& tensor, float constant, Tensor& destination);

void multiply(const Tensor& a, const Tensor& b, Tensor& destination);

void transpose(const Tensor& matrix, Tensor& destination);

#endif //NNLIB_TENSOR_H
