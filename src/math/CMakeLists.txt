set(SOURCE_FILES convert.cpp convert.h
        matrix.cpp ../../include/matrix.h
        matrix_operations.cu matrix_operations.cuh
        vector.cpp ../../include/vector.h
        vector_operations.cu vector_operations.cuh)

target_sources(nnlib PRIVATE ${SOURCE_FILES})
