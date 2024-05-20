/**
 * @file tensor.h
 * @brief Header file declaring the Tensor class to represent multidimensional arrays.
 * @author Jan Warchocki
 * @date 26 August 2022
 */

#ifndef NNLIB_TENSOR_H
#define NNLIB_TENSOR_H

#include "allocation.h"
#include "allocation_gpu.cuh"
#include "cache.h"
#include "exceptions.h"
#include "runtime.h"
#include "session.cuh"
#include <cstdlib>
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>


// forward definitions to avoid circular dependencies
template<typename T>
class Tensor;

template<typename T>
void fill(float value, std::shared_ptr<Tensor<T>>& tensor);
namespace no_grad {
    template<typename T>
    std::shared_ptr<Tensor<T>> add(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b);
}

template<typename T>
class BackwardFunction {
public:
    std::vector<std::shared_ptr<Tensor<T>>> parents;

    virtual std::vector<std::shared_ptr<Tensor<T>>> backward(std::shared_ptr<Tensor<T>> grad) = 0;

    virtual ~BackwardFunction() = default;
};

/**
 * @brief Class to represent multidimensional arrays.
 */
template<typename T>
class Tensor {

    /**
     * @brief Store the shape of the tensor.
     */
public:
    std::vector<size_t> shape;

    /**
     * @brief Store the total size of the tensor.
     *
     * The size is equal to the number of elements that can be contained in the tensor. It is computed in
     * the Tensor::computeSize() method.
     */
    size_t size;

    /**
     * @brief The location of the tensor.
     *
     * Can be either HOST or DEVICE. See #DataLocation for more details.
     */
    DataLocation location;

    /**
     * @brief The data stored by the tensor.
     */
    T* data;

    /**
     * @brief Session object containing information about current session.
     */
    Session session;

    bool requiresGrad;
    std::shared_ptr<BackwardFunction<T>> gradFunction;
    std::shared_ptr<Tensor<T>> grad;

    /**
     * @brief Initialize an empty tensor.
     */
    Tensor() : shape(), size(0), location(HOST), data(), requiresGrad(false), gradFunction(), grad() {}

    Tensor(std::vector<size_t> shape, DataLocation location) : shape(std::move(shape)), location(location), size(0), data(), requiresGrad(false), gradFunction(), grad() {
        computeSize();
        Cache& cache = Cache::getInstance();
        data = cache.get(size, location);
    }

    /**
     * @brief Tensor constructor with shape given directly by a vector.
     *
     * @param shape The shape of the vector to create.
     */
    explicit Tensor(std::vector<size_t> shape) : Tensor(std::move(shape), HOST) {}

    /**
     * @brief The copy constructor.
     *
     * @param other The tensor based on which this one should be initialized.
     */
    Tensor(const Tensor<T>& other) {
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

    /**
     * @brief Construct a tensor based on the passed shape.
     *
     * For example `%Tensor(2, 2, 3)` will initialize a 3D tensor with two 2x3 matrices.
     *
     * @param args The shapes of consecutive dimensions.
     */
    template<typename... Args>
    explicit Tensor(Args... args) : Tensor(std::vector<size_t>({static_cast<size_t>(args)...})) {}

    /**
     * @brief The assignment operator.
     *
     * This releases all the current memory and copies all the information of the @p other tensor.
     *
     * @param other The other tensor to assign this one to.
     * @return Assigned tensor. Always returns *this.
     */
    Tensor& operator=(const Tensor& other) {
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

    [[nodiscard]] std::shared_ptr<Tensor<T>> copy() const {
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
     * @brief Move the tensor to the designated destination.
     *
     * This involves copying the data to the new location and releasing memory from the old location.
     *
     * @param target The destination to move the tensor to.
     */
    void move(DataLocation target) {
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

    void useGrad() {
        requiresGrad = true;
        grad = std::make_shared<Tensor>(shape, location);
        fill(0.0f, grad);
        gradFunction = nullptr;
    }

    // NOLINTNEXTLINE(google-readability-function-size)
    void backward() {
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

    /**
     * @brief Static method to easily initialize a 1D tensor with given data.
     *
     * @param data The data based on which a tensor should be constructed.
     * @return The constructed tensor.
     */
    static Tensor construct1d(const std::vector<float>& data) {
        if (data.empty()) {
            throw SizeMismatchException();
        }
        Tensor result = Tensor(data.size());
        std::copy(data.begin(), data.end(), result.data);
        return result;
    }

    /**
     * @brief Static method to easily initialize a 2D tensor with given data.
     *
     * @param data The data based on which a tensor should be constructed.
     * @return The constructed tensor.
     */
    static Tensor construct2d(const std::vector<std::vector<float>>& data) {
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

    /**
     * @brief Method to access an element of the tensor at a specific method.
     *
     * This method should not be used in performance critical operations. In these cases a direct access
     * to Tensor::data will be more appropriate (as it becomes easier for the compiler to optimize such code).
     *
     * @param args The index of the element to access.
     * @return A reference to the requested element.
     */
    template<typename... Args>
    float& operator()(Args... args) {
        std::vector<size_t> index = std::vector<size_t>({static_cast<size_t>(args)...});
        // Make sure the indexes are within acceptable range and throw SizeMismatchException if not.
        verifyIndex(index);
        // Recursively figure out the index in the flattened array (the effective index)
        size_t effectiveIndex = findEffectiveAddress(index, shape.size() - 1);
        return data[effectiveIndex];
    }

    /**
     * @brief The destructor.
     */
    ~Tensor() {
        if (size == 0) {
            return;
        }
        Cache& cache = Cache::getInstance();
        cache.put(size, data, location);
    }

    /**
     * @brief Compute the total size of the tensor based on its shape.
     */
private:
    void computeSize() {
        size = 1;
        for (auto it = shape.begin(); it < shape.end(); it++) {
            size *= *it;
        }
    }

    /**
     * @brief Finds the address of an element in the flattened data array given its index in non-flattened tensor.
     *
     * Since the method works recursively, it also takes the @p depth parameter, which provides information
     * about which dimension is currently taken into account when computing the index.
     *
     * @param index The multidimensional index of the element to access.
     * @param depth The dimension that is currently considered in the recursive call.
     * @return The address of the element in the flattened data array.
     */
    [[nodiscard]] size_t findEffectiveAddress(const std::vector<size_t>& index, size_t depth) const {
        if (depth == 0) {
            return index.front();
        }

        return shape.at(depth) * findEffectiveAddress(index, depth - 1) + index.at(depth);
    }

    /**
     * @brief Verify that an index of an element is within the shape of the tensor.
     *
     * @param index The index of the element that is being accessed.
     */
    void verifyIndex(const std::vector<size_t>& index) const {
        if (index.size() != shape.size()) {
            throw SizeMismatchException();
        }
        for (size_t i = 0; i < index.size(); i++) {
            if (index[i] >= shape[i]) {
                throw SizeMismatchException();
            }
        }
    }

    bool canBackPropagate(const Tensor<T>& tensor) {
        return !(tensor.shape.size() != 1 || tensor.shape[0] != 1 || !tensor.requiresGrad);
    }
};

typedef std::shared_ptr<Tensor<float>> sfTensor;
typedef std::shared_ptr<Tensor<double>> sdTensor;

/**
 * @brief Enables the tensor to be printed using std::cout.
 *
 * @param stream The stream to print the tensor to.
 * @param tensor The tensor to print.
 * @return The stream with the string representation of the tensor added to it.
 */
template<typename T>
std::ostream& operator<<(std::ostream& stream, const Tensor<T>& tensor) {
    stream << "Tensor: ";
    for (int i = 0; i < tensor.size; i++) {
        stream << tensor.data[i] << " ";
    }
    return stream;
}



#endif //NNLIB_TENSOR_H
