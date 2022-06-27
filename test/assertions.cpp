//
// Created by Jan Warchocki on 27/06/2022.
//

#include "assertions.h"

::testing::AssertionResult assertEqual(const Matrix& result, std::initializer_list<std::initializer_list<DTYPE>> expected) {
    size_t expectedNumRows = expected.size();
    size_t expectedNumColumns = expected.begin()->size();

    if (result.n != expectedNumRows || result.m != expectedNumColumns) {
        return ::testing::AssertionFailure() << "Wrong shape of result matrix. Expected "
                                             << expectedNumRows << "x" << expectedNumColumns << " got "
                                             << result.n << "x" << result.m;
    }

    int i = 0; int j = 0;
    for (auto& row : expected) {
        if (row.size() != expectedNumColumns) {
            return ::testing::AssertionFailure() << "Not a valid matrix was passed as the expected result.";
        }

        for (auto value : row) {
            DTYPE actual = result(i, j++);
            if (value != actual) {
                return ::testing::AssertionFailure() << "Different matrices at index [" << i << ", "
                                                     << j - 1 << "]. Expected " << value
                                                     << " instead got " << actual;
            }
        }
        i++;
        j = 0;
    }

    return ::testing::AssertionSuccess();
}

::testing::AssertionResult assertEqual(const Vector& result, std::initializer_list<DTYPE> expected) {
    if(result.n != expected.size()) {
        return testing::AssertionFailure() << "Shape of vector invalid. Expected " << expected.size()
                                           << " entries, instead got " << result.n << " entries";
    }

    int i = 0;
    for (auto value : expected) {
        DTYPE actual = result[i++];
        if (actual != value) {
            return ::testing::AssertionFailure() << "Different vectors at index " << i - 1 << ". Expected "
                                                 << value << " instead got " << actual;
        }
    }

    return ::testing::AssertionSuccess();
}
