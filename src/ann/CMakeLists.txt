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

        backpropagation.cpp backpropagation.h
        backpropagation_on_host.cpp
        backpropagation_on_device.cu

        layer.cpp
        network.cpp)

target_sources(nnlib PRIVATE ${SOURCE_FILES})
