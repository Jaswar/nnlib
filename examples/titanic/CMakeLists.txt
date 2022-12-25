cmake_minimum_required(VERSION 3.20)
include(CheckLanguage)

project(titanic_nnlib LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)

find_package(nnlib CONFIG REQUIRED)

check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)

    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    # Add flag to specify the architecture of the GPU (compute capability)
    if (NOT DEFINED CUDA_ARCHITECTURES)
        set_target_properties(nnlib PROPERTIES CUDA_ARCHITECTURES "50")
    endif()

    set_target_properties(nnlib PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE nnlib)
