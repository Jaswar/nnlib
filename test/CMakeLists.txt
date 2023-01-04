set(NNLIB_TEST_EXEC test_nnlib)

add_executable(${NNLIB_TEST_EXEC}
        assertions.h assertions.cpp
        test_utils.h test_utils.cpp)

add_subdirectory(math)
add_subdirectory(ann)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(RC_ENABLE_GTEST ON CACHE BOOL "Rapidcheck GTest Support" FORCE)
FetchContent_Declare(
        rapidcheck
        URL https://github.com/emil-e/rapidcheck/archive/ce2b602bbe89e9264fb8725cc3f04049951f29cb.zip
)
FetchContent_MakeAvailable(rapidcheck)

enable_testing()

target_link_libraries(${NNLIB_TEST_EXEC} PUBLIC ${PROJECT_NAME} gtest_main rapidcheck rapidcheck_gtest)

include(GoogleTest rapidcheck)
gtest_discover_tests(${NNLIB_TEST_EXEC})
