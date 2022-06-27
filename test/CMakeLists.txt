set(NNLIB_TEST_EXEC test_nnlib)

add_executable(${NNLIB_TEST_EXEC}
        utils.h utils.cpp
        vector_operations_host_tests.cpp vector_operations_device_tests.cpp)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

target_link_libraries(${NNLIB_TEST_EXEC} ${PROJECT_NAME} gtest_main)

include(GoogleTest)
gtest_discover_tests(${NNLIB_TEST_EXEC})
