/**
 * @file tensor.h
 * @brief Header file declaring the Tensor class to represent multidimensional arrays.
 * @author Jan Warchocki
 * @date 26 August 2022
 */

#ifndef NNLIB_TENSOR_H
#define NNLIB_TENSOR_H

#include "allocation.h"
#include "session.cuh"
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

/**
 * @brief Enumerate to specify where data is located.
 *
 * Can be either HOST or DEVICE. In case it is set to HOST, the data is stored in RAM and is processed by the CPU.
 * In case it is set to DEVICE, the data is in VRAM and processed by the GPU. The latter is only possible if CUDA
 * is installed and there is a CUDA enabled GPU on the system.
 */
enum DataLocation { HOST, DEVICE };

/**
 * @brief Class to represent multidimensional arrays.
 */
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
    float* data;

    /**
     * @brief Session object containing information about current session.
     */
    Session session;

    /**
     * @brief Initialize an empty tensor.
     */
    Tensor();

    /**
     * @brief The copy constructor.
     *
     * @param other The tensor based on which this one should be initialized.
     */
    Tensor(const Tensor& other);

    explicit Tensor(std::vector<size_t> shape) : shape(std::move(shape)), location(HOST), size(0), data() {
        computeSize();
        data = allocate1DArray(size, 0);
    }

    /**
     * @brief Construct a tensor based on the passed shape.
     *
     * For example `%Tensor(2, 2, 3)` will initialize a 3D tensor with two 2x3 matrices.
     *
     * @param args The shapes of consecutive dimensions.
     */
    template<typename... Args>
    explicit Tensor(Args... args) : shape({static_cast<size_t>(args)...}), location(HOST), size(0), data() {
        computeSize();
        data = allocate1DArray(size, 0);
    }

    /**
     * @brief The assignment operator.
     *
     * This releases all the current memory and copies all the information of the @p other tensor.
     *
     * @param other The other tensor to assign this one to.
     * @return Assigned tensor. Always returns *this.
     */
    Tensor& operator=(const Tensor& other);

    /**
     * @brief Move the tensor to the designated destination.
     *
     * This involves copying the data to the new location and releasing memory from the old location.
     *
     * @param target The destination to move the tensor to.
     */
    void move(DataLocation target);

    /**
     * @brief Static method to easily initialize a 1D tensor with given data.
     *
     * @param data The data based on which a tensor should be constructed.
     * @return The constructed tensor.
     */
    static Tensor construct1d(const std::vector<float>& data);

    /**
     * @brief Static method to easily initialize a 2D tensor with given data.
     *
     * @param data The data based on which a tensor should be constructed.
     * @return The constructed tensor.
     */
    static Tensor construct2d(const std::vector<std::vector<float>>& data);

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
    ~Tensor();

    /**
     * @brief Compute the total size of the tensor based on its shape.
     */
private:
    void computeSize();

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
    size_t findEffectiveAddress(const std::vector<size_t>& index, size_t depth) const;

    /**
     * @brief Verify that an index of an element is within the shape of the tensor.
     *
     * @param index The index of the element that is being accessed.
     */
    void verifyIndex(const std::vector<size_t>& index) const;
};

/**
 * @brief Enables the tensor to be printed using std::cout.
 *
 * @param stream The stream to print the tensor to.
 * @param tensor The tensor to print.
 * @return The stream with the string representation of the tensor added to it.
 */
std::ostream& operator<<(std::ostream& stream, const Tensor& tensor);

/**
 * @brief Fill a tensor with a specific value.
 *
 * @param value The value to fill the tensor with.
 * @param destination The tensor to fill.
 */
void fill(float value, Tensor& destination);

/**
 * @brief Add two tensors together.
 *
 * If the first tensor is a matrix and the second a vector, the operation performed is broadcast-add.
 * See addBroadcast() for more details.
 *
 * @param a The first tensor.
 * @param b The second tensor.
 * @param destination Where to store the result of addition.
 */
void add(const Tensor& a, const Tensor& b, Tensor& destination);

/**
 * @brief Subtract one tensor from another.
 *
 * @param a The tensor to subtract from.
 * @param b The tensor to be subtracted.
 * @param destination Where to store the result of the subtraction.
 */
void subtract(const Tensor& a, const Tensor& b, Tensor& destination);

/**
 * @brief Perform hadamard product (element-wise multiplication) on two tensors.
 *
 * @param a The first tensor.
 * @param b The second tensor.
 * @param destination Where to store the result of the operation.
 */
void hadamard(const Tensor& a, const Tensor& b, Tensor& destination);

/**
 * @brief Multiply a tensor with a constant.
 *
 * @param tensor The tensor to multiply.
 * @param constant The constant to multiply the tensor with.
 * @param destination Where to store the result of the multiplication.
 */
void multiply(const Tensor& tensor, float constant, Tensor& destination);

/**
 * @brief Multiply one tensor with another.
 *
 * The only currently supported multiplications are matrix-matrix and matrix-vector. If tensors with different
 * shapes will be passed, UnsupportedOperationException will be thrown.
 *
 * @param a The first tensor.
 * @param b The second tensor.
 * @param destination Where the result of multiplication should be stored.
 */
void multiply(const Tensor& a, const Tensor& b, Tensor& destination);

/**
 * @brief Transpose a matrix.
 *
 * The tensor must be a 2D tensor, otherwise UnsupportedOperationException is thrown.
 *
 * @param matrix The matrix to transpose.
 * @param destination Where the result of the transpose operation should be stored.
 */
void transpose(const Tensor& matrix, Tensor& destination);

#endif //NNLIB_TENSOR_H
