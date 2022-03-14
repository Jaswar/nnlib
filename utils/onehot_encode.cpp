//
// Created by Jan Warchocki on 07/03/2022.
//


#include "onehot_encode.h"
#include <set>

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

    int n = vector.n;
    int m = static_cast<int>(unique.size());

    DTYPE* resultSpace = allocate1DArray(n * m, 0);
    Matrix result = Matrix(resultSpace, n, m);

    for (int i = 0; i < vector.n; i++) {
        DTYPE value = vector[i];
        int index = indexOf(value, unique);
        result(i, index) = 1;
    }

    return result;
}
