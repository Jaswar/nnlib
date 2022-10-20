/**
 * @file onehot_encode.cpp
 * @brief Source file containing the definition of the oneHotEncode() method.
 * @author Jan Warchocki
 * @date 07 March 2022
 */


#include "onehot_encode.h"
#include <exceptions/unsupported_operation_exception.h>
#include <set>

/**
 * @brief Return the index of a value in a set.
 *
 * @param value The value to look for.
 * @param set The set to look for @p value in.
 * @return The index of @p value in @p set or -1 if the value is not found.
 */
int indexOf(float value, const std::set<float>& set) {
    int index = 0;
    for (auto& v : set) {
        if (v == value) {
            return index;
        }
        index++;
    }
    return -1;
}

Tensor oneHotEncode(const Tensor& vector) {
    if (vector.shape.size() != 1) {
        throw UnsupportedOperationException();
    }

    std::set<float> unique;
    for (int i = 0; i < vector.shape[0]; i++) {
        unique.insert(vector.host[i]);
    }

    auto n = vector.shape[0];
    auto m = unique.size();

    Tensor result = Tensor(n, m);
    fill(0, result);

    for (int i = 0; i < vector.shape[0]; i++) {
        float value = vector.host[i];
        int index = indexOf(value, unique);
        result.host[i * result.shape[1] + index] = 1;
    }

    return result;
}
