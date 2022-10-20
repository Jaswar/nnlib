/**
 * @file tensor.h
 * @brief Header file declaring the Tensor class to represent multidimensional arrays.
 * @author Jan Warchocki
 * @date 26 August 2022
 */

#ifndef NNLIB_TENSOR_H
#define NNLIB_TENSOR_H

#include "allocation.h"
#include "session.h"
#include <cstdlib>
#include <iostream>
#include <vector>

enum DataLocation { HOST, DEVICE };

class Tensor {
public:
    std::vector<size_t> shape;
    size_t size;

    DataLocation location;
    float* data;

    Session session;

    Tensor();

    Tensor(const Tensor& other);

    template<typename... Args>
    explicit Tensor(Args... args) : shape({static_cast<size_t>(args)...}), location(HOST), size(0), data() {
        computeSize();
        data = allocate1DArray(size, 0);
    }

    Tensor& operator=(const Tensor& other);

    void move(DataLocation target);

    static Tensor construct1d(const std::vector<float>& data);
    static Tensor construct2d(const std::vector<std::vector<float>>& data);

    template<typename... Args>
    float& operator()(Args... args) {
        std::vector<size_t> index = std::vector<size_t>({static_cast<size_t>(args)...});
        // Make sure the indexes are within acceptable range and throw SizeMismatchException if not.
        verifyIndex(index);
        // Recursively figure out the index in the flattened array (the effective index)
        size_t effectiveIndex = findEffectiveAddress(index, shape.size() - 1);
        return data[effectiveIndex];
    }

    ~Tensor();

private:
    void computeSize();
    size_t findEffectiveAddress(const std::vector<size_t>& index, size_t depth) const;
    void verifyIndex(const std::vector<size_t>& index) const;
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
