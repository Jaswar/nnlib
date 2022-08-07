/**
 * @file matrix.h
 * @brief Header file declaring the Matrix class and operations on matrices.
 *
 * All the mathematical methods (such as add, subtract, multiply), have implementations for both CPU and GPU.
 * The correct implementation is called depending on the locations of the operands. If not all operands are
 * located in the same place, the methods throw DifferentDataLocationException.
 *
 * Furthermore, all mathematical methods require a pre-allocated Matrix or Vector, where the result should
 * be written to. This is to minimize allocating memory and put more effort into pre-allocating space and reusing it.
 *
 * @author Jan Warchocki
 * @date 03 March 2022
 */

#ifndef NNLIB_MATRIX_H
#define NNLIB_MATRIX_H

#include "vector.h"

/**
 * @brief Represents a matrix (2D array).
 */
class Matrix {

    /**
     * @brief The data stored by the matrix.
     *
     * Although the matrix is a 2D array, the data is stored in 1D format. This allows for easier GPU integration.
     */
public:
    DTYPE* data;

    /**
     * @brief The number of rows of the matrix.
     */
    size_t n;

    /**
     * @brief The number of columns of the matrix.
     */
    size_t m;

    /**
     * @brief The location of Matrix::data.
     *
     * Can be either HOST or DEVICE. See #DataLocation for more information.
     */
    DataLocation location;

    /**
     * @brief Construct the matrix using only size of the matrix.
     *
     * By default, the matrix is initialized on host memory. The data is allocated in the constructor.
     *
     * @param n The number of rows of the matrix.
     * @param m The number of columns of the matrix.
     */
    Matrix(size_t n, size_t m);

    /**
     * @brief Construct the matrix using size and location as specified.
     *
     * @param n The number of rows of the matrix.
     * @param m The number of columns of the matrix.
     * @param location The destined location of the matrix.
     */
    Matrix(size_t n, size_t m, DataLocation location);

    /**
     * @brief Construct matrix using existing data and size.
     *
     * Assumes the data is stored on host.
     *
     * @param data The already-allocated on-host data.
     * @param n The number of rows of the matrix.
     * @param m The number of columns of the matrix.
     */
    Matrix(DTYPE* data, size_t n, size_t m);

    /**
     * @brief Construct a matrix using all attributes.
     *
     * @param data The already-allocated data.
     * @param n The number of rows of the matrix.
     * @param m The number of columns of the matrix.
     * @param location The location of data.
     */
    Matrix(DTYPE* data, size_t n, size_t m, DataLocation location);

    /**
     * @brief The copy constructor.
     *
     * @param matrix The other matrix to copy into this matrix.
     */
    Matrix(const Matrix& matrix);

    /**
     * @brief Move the data of the matrix to device.
     */
    void moveToDevice();

    /**
     * @brief Move the data of the matrix to host.
     */
    void moveToHost();

    /**
     * @brief The destructor of a matrix.
     *
     * Ensures that correct method is called depending if the location of the matrix is host or device.
     */
    ~Matrix();

    /**
     * @brief The assignment operator.
     *
     * Frees the current memory of the matrix and copies the data taken from @p matrix.
     *
     * @param matrix The other matrix to assign this one into.
     * @return Assigned matrix. Always returns `*this`.
     */
    Matrix& operator=(const Matrix& matrix);

    /**
     * @brief Access a specific element of the matrix.
     *
     * Returns a reference to the element, so that it can be modified using this method.
     *
     * @param x The row index of the element to access.
     * @param y The column index of the element to access.
     * @return Reference to the requested element.
     */
    DTYPE& operator()(size_t x, size_t y) const;
};

/**
 * @brief Method that allows for printing the matrix using std::cout.
 *
 * @param stream The std::cout stream.
 * @param matrix The matrix to print.
 * @return Stream with the string representation of the matrix added to it.
 */
std::ostream& operator<<(std::ostream& stream, const Matrix& matrix);

/**
 * @brief Add two matrices together.
 *
 * @param m1 The first matrix.
 * @param m2 The second matrix.
 * @param result The result of adding @p m1 and @p m2.
 *
 * @throws SizeMismatchException If the parameter matrices are different shapes.
 * @throws DifferentDataLocationException If not all parameters are located in the same place.
 */
void add(const Matrix& m1, const Matrix& m2, Matrix& result);

/**
 * @brief Perform broadcast operation to add a vector to a matrix.
 *
 * The addition is performed along the first axis. As a result, @p v is added to every row of @p m.
 *
 * @param m The matrix.
 * @param v The vector.
 * @param result The resulting matrix with @p v broadcast onto the rows of @p m and added.
 *
 * @throws SizeMismatchException If the number of columns in @p m is not the same as size of @p v or if
 * @p result has a different shape than @p m.
 * @throws DifferentDataLocationException If not all parameters are located in the same place.
 */
void add(const Matrix& m, const Vector& v, Matrix& result);

/**
 * @brief Subtract one matrix from another.
 *
 * @param m1 The first matrix.
 * @param m2 The second matrix.
 * @param result The result of subtracting @p m2 from @p m1.
 *
 * @throws SizeMismatchException If the parameter matrices are different shapes.
 * @throws DifferentDataLocationException If not all parameters are located in the same place.
 */
void subtract(const Matrix& m1, const Matrix& m2, Matrix& result);

/**
 * @brief Multiply one matrix with another.
 *
 * @param m1 The first matrix.
 * @param m2 The second matrix.
 * @param result The result of multiplying @p m1 with @p m2.
 *
 * @throws SizeMismatchException If the parameter matrices are in shapes that make multiplication impossible.
 * @throws DifferentDataLocationException If not all parameters are located in the same place.
 */
void multiply(const Matrix& m1, const Matrix& m2, Matrix& result);

/**
 * @brief Multiply a matrix with a vector.
 *
 * @param m The matrix.
 * @param v The vector.
 * @param result The result of multiplying @p m with @p v.
 *
 * @throws SizeMismatchException If number of columns of @p m is different than size of @p v or if sizes of @p v and
 * @p result are not the same.
 * @throws DifferentDataLocationException If not all parameters are located in the same place.
 */
void multiply(const Matrix& m, const Vector& v, Vector& result);

/**
 * @brief Multiply a matrix with a #DTYPE constant.
 *
 * @param m The matrix to multiply.
 * @param constant The constant to multiply @p m with.
 * @param result The result of multiplication.
 *
 * @throws SizeMismatchException If @p m and @p result are not the same shape.
 * @throws DifferentDataLocationException If @p m and @p result are not located in the same place.
 */
void multiply(const Matrix& m, DTYPE constant, Matrix& result);

/**
 * @brief Multiply a matrix with a #DTYPE constant.
 *
 * @param constant The constant to multiply @p m with.
 * @param m The matrix to multiply.
 * @param result The result of multiplication.
 *
 * @throws SizeMismatchException If @p m and @p result are not the same shape.
 * @throws DifferentDataLocationException If @p m and @p result are not located in the same place
 */
void multiply(DTYPE constant, const Matrix& m, Matrix& result);

/**
 * @brief Perform hadamard product (element-wise multiplication) on two matrices.
 *
 * @param m1 The first matrix.
 * @param m2 The second matrix.
 * @param result The result of performing the hadamard product.
 *
 * @throws SizeMismatchException If the parameter matrices are different shapes.
 * @throws DifferentDataLocationException If not all parameters are located in the same place.
 */
void hadamard(const Matrix& m1, const Matrix& m2, Matrix& result);

/**
 * @brief Transpose a matrix.
 *
 * @param m The matrix to transpose.
 * @param result The transposed matrix.
 *
 * @throws SizeMismatchException If shape of @p result is not the transpose of @p m.
 * @throws DifferentDataLocationException If not all parameters are located in the same place.
 */
void transpose(const Matrix& m, Matrix& result);

#endif //NNLIB_MATRIX_H
