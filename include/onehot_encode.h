/**
 * @file onehot_encode.h
 * @brief Header file containing the declaration of the oneHotEncode() method.
 * @author Jan Warchocki
 * @date 07 March 2022
 */

#ifndef NNLIB_ONEHOT_ENCODE_H
#define NNLIB_ONEHOT_ENCODE_H

#include "tensor.h"

/**
 * @brief One hot encode a vector of data.
 *
 * @param vector The vector to encode.
 * @return %Matrix corresponding to the one hot encoded vector.
 */
Tensor oneHotEncode(const Tensor& vector);

#endif //NNLIB_ONEHOT_ENCODE_H
