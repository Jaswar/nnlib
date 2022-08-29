set(SOURCE_FILES convert.cpp

        tensor.cpp
        tensor_operations_on_host.h tensor_operations_on_host.cpp
        tensor_operations_on_device.cuh tensor_operations_on_device.cu

        matrix/matrix.cpp
        matrix/matrix_operations_on_host.cpp matrix/matrix_operations_on_host.h
        matrix/matrix_operations_on_device.cu matrix/matrix_operations_on_device.cuh

        vector/vector.cpp
        vector/vector_operations_on_host.cpp vector/vector_operations_on_host.h
        vector/vector_operations_on_device.cu vector/vector_operations_on_device.cuh)

target_sources(nnlib PRIVATE ${SOURCE_FILES})
