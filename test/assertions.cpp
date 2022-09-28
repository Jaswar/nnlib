//
// Created by Jan Warchocki on 27/06/2022.
//

#include "assertions.h"

::testing::AssertionResult assertEqual(const Tensor& result,
                                       std::initializer_list<std::initializer_list<DTYPE>> expected) {
    size_t expectedNumRows = expected.size();
    size_t expectedNumColumns = expected.begin()->size();

    if (result.shape[0] != expectedNumRows || result.shape[1] != expectedNumColumns) {
        return ::testing::AssertionFailure() << "Wrong shape of result matrix. Expected " << expectedNumRows << "x"
                                             << expectedNumColumns << " got " << result.shape[0] << "x" << result.shape[1];
    }

    int i = 0;
    int j = 0;
    for (auto& row : expected) {
        if (row.size() != expectedNumColumns) {
            return ::testing::AssertionFailure() << "Not a valid matrix was passed as the expected result.";
        }

        for (auto value : row) {
            DTYPE actual = result.host[i * result.shape[1] + j++];
            if (value != actual) {
                return ::testing::AssertionFailure() << "Different matrices at index [" << i << ", " << j - 1
                                                     << "]. Expected " << value << " instead got " << actual;
            }
        }
        i++;
        j = 0;
    }

    return ::testing::AssertionSuccess();
}

::testing::AssertionResult assertEqual(const Tensor& result, std::initializer_list<DTYPE> expected) {
    if (result.shape[0] != expected.size()) {
        return testing::AssertionFailure() << "Shape of vector invalid. Expected " << expected.size()
                                           << " entries, instead got " << result.shape[0] << " entries";
    }

    int i = 0;
    for (auto value : expected) {
        DTYPE actual = result.host[i++];
        if (actual != value) {
            return ::testing::AssertionFailure()
                   << "Different vectors at index " << i - 1 << ". Expected " << value << " instead got " << actual;
        }
    }

    return ::testing::AssertionSuccess();
}

::testing::AssertionResult assertClose(const Tensor& result, std::initializer_list<float> expected, float delta) {
    if (result.shape[0] != expected.size()) {
        return testing::AssertionFailure() << "Shape of vector invalid. Expected " << expected.size()
                                           << " entries, instead got " << result.shape[0] << " entries";
    }

    int i = 0;
    for (auto value : expected) {
        DTYPE actual = result.host[i++];
        if (std::abs(actual - value) > delta) {
            return ::testing::AssertionFailure()
                   << "Different vectors at index " << i - 1 << ". Expected " << value << " instead got " << actual;
        }
    }

    return ::testing::AssertionSuccess();
}

::testing::AssertionResult assertClose(const Tensor& result,
                                       std::initializer_list<std::initializer_list<float>> expected, float delta) {
    size_t expectedNumRows = expected.size();
    size_t expectedNumColumns = expected.begin()->size();

    if (result.shape[0] != expectedNumRows || result.shape[1] != expectedNumColumns) {
        return ::testing::AssertionFailure() << "Wrong shape of result matrix. Expected " << expectedNumRows << "x"
                                             << expectedNumColumns << " got " << result.shape[0] << "x" << result.shape[1];
    }

    int i = 0;
    int j = 0;
    for (auto& row : expected) {
        if (row.size() != expectedNumColumns) {
            return ::testing::AssertionFailure() << "Not a valid matrix was passed as the expected result.";
        }

        for (auto value : row) {
            DTYPE actual = result.host[i * result.shape[1] + j++];
            if (std::abs(value - actual) > delta) {
                return ::testing::AssertionFailure() << "Different matrices at index [" << i << ", " << j - 1
                                                     << "]. Expected " << value << " instead got " << actual;
            }
        }
        i++;
        j = 0;
    }

    return ::testing::AssertionSuccess();
}
