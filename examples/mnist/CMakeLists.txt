cmake_minimum_required(VERSION 3.20)

project(mnist_nnlib LANGUAGES CUDA CXX)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_PREFIX_PATH "C:/Users/janwa/CLionProjects/nnlib/install")
find_package(nnlib CONFIG REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE nnlib::nnlib)

if (NOT DEFINED CUDA_ARCHITECTURES)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "50")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

