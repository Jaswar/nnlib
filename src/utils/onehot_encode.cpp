/**
 * @file onehot_encode.cpp
 * @brief Source file containing the definition of the oneHotEncode() method.
 * @author Jan Warchocki
 * @date 07 March 2022
 */


#include "onehot_encode.h"
#include <set>

/**
 * @brief Return the index of a value in a set.
 *
 * @param value The value to look for.
 * @param set The set to look for @p value in.
 * @return The index of @p value in @p set or -1 if the value is not found.
 */
int indexOf(DTYPE value, const std::set<DTYPE>& set) {
    int index = 0;
    for (auto& v : set) {
        if (v == value) {
            return index;
        }
        index++;
    }
    return -1;
}

Matrix oneHotEncode(const Vector& vector) {
    std::set<DTYPE> unique;
    for (int i = 0; i < vector.n; i++) {
        unique.insert(vector[i]);
    }

    auto n = vector.n;
    auto m = unique.size();

    DTYPE* resultSpace = allocate1DArray(n * m, 0);
    Matrix result = Matrix(resultSpace, n, m);

    for (int i = 0; i < vector.n; i++) {
        DTYPE value = vector[i];
        int index = indexOf(value, unique);
        result(i, index) = 1;
    }

    return result;
}
