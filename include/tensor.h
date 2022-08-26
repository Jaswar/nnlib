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

    template<typename... Args>
    explicit Tensor(Args... args) : shape({static_cast<size_t>(args)...}), location(HOST), size(0), device(nullptr) {
        computeSize();
        host = allocate1DArray(size);
    }

    void moveToDevice();
    void moveToHost();

private:
    void computeSize();

};




#endif //NNLIB_TENSOR_H
