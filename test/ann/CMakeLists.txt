set(TEST_FILES
        losses/mean_squared_error.cpp
        losses/binary_cross_entropy.cpp
        losses/categorical_cross_entropy.cpp

        metrics/categorical_accuracy.cpp
        metrics/binary_accuracy.cpp

        network_test.cpp)

target_sources(${NNLIB_TEST_EXEC} PRIVATE ${TEST_FILES})