set(SOURCE_FILES different_data_location_exception.cpp different_data_location_exception.h
        size_mismatch_exception.cpp size_mismatch_exception.h
        unexpected_cuda_call_exception.cpp unexpected_cuda_call_exception.h
        unsupported_operation_exception.cpp unsupported_operation_exception.h)

target_sources(nnlib PRIVATE ${SOURCE_FILES})
