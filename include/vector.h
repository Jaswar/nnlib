/**
 * @file vector.h
 * @brief Header file declaring the Vector class and operations on vectors.
 *
 * All the mathematical methods (such as add, subtract, multiply), have implementations for both CPU and GPU.
 * The correct implementation is called depending on the locations of the operands. If not all operands are
 * located in the same place, the methods throw DifferentDataLocationException.
 *
 * Furthermore, all mathematical methods require a pre-allocated Vector, where the result should
 * be written to. This is to minimize allocating memory and put more effort into pre-allocating space and reusing it.
 *
 * @author Jan Warchocki
 * @date 03 March 2022
 */

#ifndef NNLIB_VECTOR_H
#define NNLIB_VECTOR_H

#include "allocation.h"
#include <iostream>

/**
 * @brief Enumerate to specify where data is located.
 *
 * Can be either HOST or DEVICE. In case it is set to HOST, the data is stored in RAM and is processed by the CPU.
 * In case it is set to DEVICE, the data is in VRAM and processed by the GPU. The latter is only possible if CUDA
 * is installed and there is a CUDA enabled GPU on the system.
 */
enum DataLocation { HOST, DEVICE };

/**
 * @brief Represents a vector (1D array).
 */
class Vector {

    /**
     * @brief The data stored by the vector.
     */
public:
    DTYPE* data;

    /**
     * @brief The size of the vector.
     */
    size_t n;

    /**
     * @brief The location of Vector::data.
     *
     * Can be either HOST or DEVICE. See #DataLocation for more information.
     */
    DataLocation location;

    /**
     * @brief Construct the vector using only the size.
     *
     * By default, the vector is initialized on host memory. The data is allocated in the constructor.
     *
     * @param n The size of the vector.
     */
    explicit Vector(size_t n);

    /**
     * @brief Construct the vector using size and destined location.
     *
     * @param n The size of the vector.
     * @param location The location where the vector should be stored.
     */
    Vector(size_t n, DataLocation location);

    /**
     * @brief Construct the vector using size and existing data.
     *
     * The constructor assumes the data is on host.
     *
     * @param data The already-allocated on-host data.
     * @param n The size of the vector.
     */
    Vector(DTYPE* data, size_t n);

    /**
     * @brief Construct the vector using all attributes.
     *
     * @param data The pre-allocated data.
     * @param n The size of the vector.
     * @param location The location of data.
     */
    Vector(DTYPE* data, size_t n, DataLocation location);

    /**
     * @brief The copy constructor.
     *
     * @param vector The vector that should be copied into this vector.
     */
    Vector(const Vector& vector);

    /**
     * @brief The destructor of the vector.
     *
     * Ensures the correct method is called to free the memory depending on if the data is located on host or device.
     */
    ~Vector();

    /**
     * @brief Move the data of the vector to device memory.
     */
    void moveToDevice();

    /**
     * @brief Move the data of the vector to host memory.
     */
    void moveToHost();

    /**
     * @brief The assignment operator.
     *
     * Frees the current memory of the vector and copies the data taken from @p other.
     *
     * @param other The other vector to assign this one to.
     * @return Assigned vector. Always returns *this.
     */
    Vector& operator=(const Vector& other);

    /**
     * @brief Access a specific element of the vector.
     *
     * Returns a reference to the element, so that the element can be modified.
     *
     * @param index The index of the element to access.
     * @return A reference to the requested element.
     */
    DTYPE& operator[](size_t index) const;
};

/**
 * @brief Method that allows for printing the vector using std::cout.
 *
 * @param stream The std::cout stream.
 * @param vector The vector to print.
 * @return Stream with the string representation of the vector added to it.
 */
std::ostream& operator<<(std::ostream& stream, const Vector& vector);

/**
 * @brief Add two vectors together.
 *
 * @param v1 The first vector.
 * @param v2 The second vector.
 * @param result The result of adding @p v1 and @p v2.
 *
 * @throws SizeMismatchException If parameter vectors are different size.
 * @throws DifferentDataLocationException If parameter vectors are located in different places.
 */
void add(const Vector& v1, const Vector& v2, Vector& result);

/**
 * @brief Subtract one vector from another.
 *
 * @param v1 The first vector.
 * @param v2 The second vector.
 * @param result The result of subtracting @p v2 from @p v1.
 *
 * @throws SizeMismatchException If parameter vectors are different size.
 * @throws DifferentDataLocationException If parameter vectors are located in different places.
 */
void subtract(const Vector& v1, const Vector& v2, Vector& result);

/**
 * @brief Multiply a vector with a constant.
 *
 * @param v1 The vector to be multiplied.
 * @param constant The constant to multiply the vector with.
 * @param result The result of multiplying @p v1 by @p constant.
 *
 * @throws SizeMismatchException If parameter vectors are different size.
 * @throws DifferentDataLocationException If parameter vectors are located in different places.
 */
void multiply(const Vector& v1, DTYPE constant, Vector& result);

/**
 * @brief Multiply a vector with a constant.
 *
 * @param constant The constant to multiply the vector with.
 * @param v1 The vector to be multiplied.
 * @param result The result of multiplying @p v1 by @p constant.
 *
 * @throws SizeMismatchException If parameter vectors are different size.
 * @throws DifferentDataLocationException If parameter vectors are located in different places.
 */
void multiply(DTYPE constant, const Vector& v1, Vector& result);

#endif //NNLIB_VECTOR_H
