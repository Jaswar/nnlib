set(SOURCE_FILES
        tensor.cpp
        tensor_operations_on_host.h tensor_operations_on_host.cpp
        tensor_operations_on_device.cuh tensor_operations_on_device.cu
        functions.cpp)

target_sources(nnlib PRIVATE ${SOURCE_FILES})
