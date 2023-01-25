set(TEST_FILES
        tensor_operations_common.h
        tensor_operations_common.cpp
        tensor_operations_host.cpp
        tensor_operations_device.cpp)

target_sources(${NNLIB_TEST_EXEC} PRIVATE ${TEST_FILES})