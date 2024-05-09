set(SOURCE_FILES
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
