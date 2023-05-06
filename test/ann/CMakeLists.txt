set(TEST_FILES
        activations/linear/linear_activation_common.h
        activations/linear/linear_activation_common.cpp
        activations/linear/linear_activation_host.cpp
        activations/linear/linear_activation_device.cpp

        activations/ReLU/relu_activation_common.h
        activations/ReLU/relu_activation_common.cpp
        activations/ReLU/relu_activation_host.cpp
        activations/ReLU/relu_activation_device.cpp

        activations/sigmoid/sigmoid_activation_common.h
        activations/sigmoid/sigmoid_activation_common.cpp
        activations/sigmoid/sigmoid_activation_host.cpp
        activations/sigmoid/sigmoid_activation_device.cpp

        losses/mean_squared_error.cpp
        losses/binary_cross_entropy.cpp
        losses/categorical_cross_entropy.cpp

        metrics/categorical_accuracy.cpp

        network_test.cpp)

target_sources(${NNLIB_TEST_EXEC} PRIVATE ${TEST_FILES})