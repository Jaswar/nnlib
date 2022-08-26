set(TEST_FILES
        tensor_operations.cpp

        vector_operations_host_tests.cpp vector_operations_device_tests.cpp
        matrix_operations_host_tests.cpp matrix_operations_device_tests.cpp)

target_sources(${NNLIB_TEST_EXEC} PRIVATE ${TEST_FILES})