set(TEST_FILES
        activations/linear/linear_activation_host.cpp
        activations/linear/linear_activation_device.cpp

        activations/ReLU/relu_activation_host.cpp
        activations/ReLU/relu_activation_device.cpp

        activations/sigmoid/sigmoid_activation_host.cpp
        activations/sigmoid/sigmoid_activation_device.cpp)

target_sources(${NNLIB_TEST_EXEC} PRIVATE ${TEST_FILES})