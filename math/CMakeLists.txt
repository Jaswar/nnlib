set(SOURCE_FILES convert.cpp convert.h
        matrix.cpp matrix.h
        matrix_operations.cu matrix_operations.cuh
        vector.cpp vector.h
        vector_operations.cu vector_operations.cuh)

target_sources(nnlib PRIVATE ${SOURCE_FILES})
