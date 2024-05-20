set(SOURCE_FILES
        tensor_operations_on_device.cu
        tensor_operations_on_host.cpp
)

target_sources(nnlib PRIVATE ${SOURCE_FILES})
