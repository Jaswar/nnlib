/**
 * @file tensor.cpp
 * @brief Source file defining methods related to the Tensor class.
 * @author Jan Warchocki
 * @date 26 August 2022
 */

#include "tensor.h"
#include "../gpu/allocation_gpu.cuh"
#include "cache.h"
#include "runtime.h"
#include "tensor_operations_on_device.cuh"
#include "tensor_operations_on_host.h"
#include <exceptions.h>
#include <functions.h>
#include <memory>
#include <queue>
#include <string>
#include <utils/location_verifiers.h>

template<typename T>
Tensor<T>::Tensor() : shape(), size(0), location(HOST), data(), requiresGrad(false), gradFunction(), grad() {
}

template<typename T>
Tensor<T>::Tensor(std::vector<size_t> shape)
    : shape(std::move(shape)), location(HOST), size(0), data(), requiresGrad(false), gradFunction(), grad() {
    computeSize();
    Cache& cache = Cache::getInstance();
    data = cache.get(size, location);
}

template<typename T>
Tensor<T>::Tensor(std::vector<size_t> shape, DataLocation location)
    : shape(std::move(shape)), location(location), size(0), data(), requiresGrad(false), gradFunction(), grad() {
    computeSize();
    Cache& cache = Cache::getInstance();
    data = cache.get(size, location);
}

template<typename T>
Tensor<T>::Tensor(const Tensor<T>& other) {
    location = other.location;
    // This copies the vector
    shape = other.shape;
    size = other.size;
    requiresGrad = other.requiresGrad;
    gradFunction = other.gradFunction;
    if (other.grad != nullptr) {
        grad = std::make_shared<Tensor>(*other.grad);
    }

    if (size == 0) {
        return;
    }

    Cache& cache = Cache::getInstance();
    data = cache.get(size, other.location);
    if (location == HOST) {
        copy1DArray(size, other.data, data);
    } else {
        copy1DArrayDevice(size, other.data, data);
    }
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& other) {
    if (&other == this) {
        return *this;
    }

    Cache& cache = Cache::getInstance();
    cache.put(size, data, location); // Mark the memory as reusable

    location = other.location;
    // This copies the vector
    shape = other.shape;
    size = other.size;

    data = cache.get(size, other.location);
    if (location == HOST) {
        copy1DArray(size, other.data, data);
    } else {
        copy1DArrayDevice(size, other.data, data);
    }

    return *this;
}

template<typename T>
void Tensor<T>::move(DataLocation target) {
    if (location == target) {
        return;
    }

    Cache& cache = Cache::getInstance();
    cache.put(size, data, location); // Mark the memory as reusable
    float* newData = cache.get(size, target);
    if (location == HOST) {
        copy1DFromHostToDevice(data, newData, size);
    } else {
        copy1DFromDeviceToHost(data, newData, size);
    }
    data = newData;
    location = target;
    if (grad != nullptr) {
        grad->move(target);
    }
}

template<typename T>
Tensor<T>::~Tensor() {
    if (size == 0) {
        return;
    }
    Cache& cache = Cache::getInstance();
    cache.put(size, data, location);
}

template<typename T>
void Tensor<T>::computeSize() {
    size = 1;
    for (auto it = shape.begin(); it < shape.end(); it++) {
        size *= *it;
    }
}

template<typename T>
Tensor<T> Tensor<T>::construct1d(const std::vector<float>& data) {
    if (data.empty()) {
        throw SizeMismatchException();
    }
    Tensor result = Tensor(data.size());
    std::copy(data.begin(), data.end(), result.data);
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::construct2d(const std::vector<std::vector<float>>& data) {
    if (data.empty() || data[0].empty()) {
        throw SizeMismatchException();
    }

    Tensor result = Tensor(data.size(), data[0].size());
    for (size_t i = 0; i < data.size(); i++) {
        // Make sure the array has the same number of columns in each row
        if (data[i].size() != result.shape[1]) {
            throw SizeMismatchException();
        }

        for (size_t j = 0; j < data[0].size(); j++) {
            result.data[i * result.shape[1] + j] = data[i][j];
        }
    }
    return result;
}

template<typename T>
size_t Tensor<T>::findEffectiveAddress(const std::vector<size_t>& index, size_t depth) const {
    if (depth == 0) {
        return index.front();
    }

    return shape.at(depth) * findEffectiveAddress(index, depth - 1) + index.at(depth);
}

template<typename T>
void Tensor<T>::verifyIndex(const std::vector<size_t>& index) const {
    if (index.size() != shape.size()) {
        throw SizeMismatchException();
    }
    for (size_t i = 0; i < index.size(); i++) {
        if (index[i] >= shape[i]) {
            throw SizeMismatchException();
        }
    }
}

template<typename T>
void Tensor<T>::useGrad() {
    requiresGrad = true;
    grad = std::make_shared<Tensor>(shape, location);
    fill(0.0f, grad);
    gradFunction = nullptr;
}

template<typename T>
bool canBackPropagate(const Tensor<T>& tensor) {
    return !(tensor.shape.size() != 1 || tensor.shape[0] != 1 || !tensor.requiresGrad);
}

template<typename T>
// NOLINTNEXTLINE(google-readability-function-size)
void Tensor<T>::backward() {
    if (!canBackPropagate(*this)) {
        throw UnsupportedOperationException();
    }

    auto gradient = std::make_shared<Tensor<T>>(shape, location);
    fill(1.0f, gradient);
    auto current = std::make_shared<Tensor>(*this);

    std::queue<std::pair<std::shared_ptr<Tensor<T>>, std::shared_ptr<Tensor<T>>>> queue;
    queue.emplace(current, gradient);
    while (!queue.empty()) {
        current = queue.front().first;
        gradient = queue.front().second;
        queue.pop();

        std::shared_ptr<BackwardFunction<T>> gradFn = current->gradFunction;
        if (gradFn == nullptr) {
            if (current->requiresGrad) {
                current->grad = no_grad::add(current->grad, gradient);
            }
        } else {
            auto newGrads = gradFn->backward(gradient);
            for (int i = 0; i < newGrads.size(); i++) {
                auto parent = gradFn->parents[i];
                if (parent->requiresGrad) {
                    queue.emplace(parent, newGrads[i]);
                }
            }
        }
    }
}

template<typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::copy() const {
    auto copy = std::make_shared<Tensor<T>>(shape, location);
    copy->requiresGrad = requiresGrad;
    copy->grad = nullptr;
    if (grad != nullptr) {
        copy->grad = grad->copy();
    }
    copy->gradFunction = gradFunction;

    if (location == HOST) {
        copy1DArray(size, data, copy->data);
    } else {
        copy1DArrayDevice(size, data, copy->data);
    }

    return copy;
}

/**
 * @brief Convert the shape of the tensor to a string.
 *
 * The shape is displayed in the format: `[2, 2, 3]`.
 *
 * @param tensor The tensor whose shape to show as a string.
 * @return The string representation of the shape of the tensor.
 */
template<typename T>
std::string tensorShapeToString(const Tensor<T>& tensor) {
    std::string shapeString = "[";

    for (auto it = tensor.shape.begin(); it < tensor.shape.end(); it++) {
        shapeString += std::to_string(*it);
        if (it != tensor.shape.end() - 1) {
            shapeString += ", ";
        }
    }

    return shapeString + "]";
}

template<typename T>
std::ostream& operator<<(std::ostream& stream, const Tensor<T>& tensor) {
    stream << "Tensor: ";
    for (int i = 0; i < tensor.size; i++) {
        stream << tensor.data[i] << " ";
    }
    return stream;
    //    if (tensor.location == DEVICE) {
    //        return stream << "Tensor located on device with shape: " + tensorShapeToString(tensor);
    //    } else {
    //        return stream << "Tensor located on host with shape: " + tensorShapeToString(tensor);
    //    }
}