set(SOURCE_FILES different_data_location_exception.cpp
        size_mismatch_exception.cpp
        unexpected_cuda_call_exception.cpp
        unsupported_operation_exception.cpp)

target_sources(nnlib PRIVATE ${SOURCE_FILES})
