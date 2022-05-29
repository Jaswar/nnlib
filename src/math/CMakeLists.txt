set(SOURCE_FILES convert.cpp ../../include/convert.h
        matrix.cpp ../../include/matrix.h

        matrix_operations_on_host.cpp matrix_operations_on_host.h
        matrix_operations_on_device.cu matrix_operations_on_device.cuh

        vector.cpp ../../include/vector.h
        vector_operations.cu vector_operations.cuh)

target_sources(nnlib PRIVATE ${SOURCE_FILES})
