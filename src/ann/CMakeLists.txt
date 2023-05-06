set(SOURCE_FILES activations/activation.cpp

        activations/linear/linear_activation.cpp
        activations/linear/linear_on_host_evaluator.cpp
        activations/linear/linear_on_device_evaluator.cu

        activations/ReLU/relu_activation.cpp
        activations/ReLU/relu_on_host_evaluator.cpp
        activations/ReLU/relu_on_device_evaluator.cu

        activations/sigmoid/sigmoid_activation.cpp
        activations/sigmoid/sigmoid_on_host_evaluator.cpp
        activations/sigmoid/sigmoid_on_device_evaluator.cu

        losses/loss.cpp
        losses/mean_squared_error.cpp
        losses/binary_cross_entropy.cpp
        losses/categorical_cross_entropy.cpp

        metrics/metric.cpp
        metrics/categorical_accuracy.cpp
        metrics/binary_accuracy.cpp

        layer.cpp
        network.cpp)

target_sources(nnlib PRIVATE ${SOURCE_FILES})
