set(TEST_FILES
        linear_activation_host_tests.cpp linear_activation_device_tests.cpp
        relu_activation_host_tests.cpp relu_activation_device_tests.cpp
        sigmoid_activation_host_tests.cpp sigmoid_activation_device_tests.cpp)

target_sources(${NNLIB_TEST_EXEC} PRIVATE ${TEST_FILES})
