/**
 * @file different_data_location_exception.h
 * @brief Header file declaring the DifferentDataLocationException class.
 * @author Jan Warchocki
 * @date 14 March 2022
 */

#ifndef NNLIB_DIFFERENT_DATA_LOCATION_EXCEPTION_H
#define NNLIB_DIFFERENT_DATA_LOCATION_EXCEPTION_H


#include <exception>

/**
 * @brief Exception to be thrown where operands are located in different places.
 *
 * For example, when performing the add operation on two matrices, and one of them is located on host
 * and one on device, this exception should be thrown.
 */
class DifferentDataLocationException : public std::exception {

    /**
     * @brief Return the exception description.
     *
     * @return The exception description.
     */
    const char* what() const noexcept override;
};


#endif //NNLIB_DIFFERENT_DATA_LOCATION_EXCEPTION_H
