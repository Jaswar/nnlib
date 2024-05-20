set(SOURCE_FILES
        tensor.cpp
        tensor_operations_on_device.cu
        tensor_operations_on_host.cpp
)

target_sources(nnlib PRIVATE ${SOURCE_FILES})
