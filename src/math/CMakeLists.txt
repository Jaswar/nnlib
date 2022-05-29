set(SOURCE_FILES convert.cpp ../../include/convert.h

        matrix/matrix.cpp ../../include/matrix.h
        matrix/matrix_operations_on_host.cpp matrix/matrix_operations_on_host.h
        matrix/matrix_operations_on_device.cu matrix/matrix_operations_on_device.cuh

        vector/vector.cpp ../../include/vector.h
        vector/vector_operations_on_host.cpp vector/vector_operations_on_host.h
        vector/vector_operations_on_device.cu vector/vector_operations_on_device.cuh)

target_sources(nnlib PRIVATE ${SOURCE_FILES})
