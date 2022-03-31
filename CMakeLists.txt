cmake_minimum_required(VERSION 3.20)
include(CheckLanguage)

project(nnlib)

set(CMAKE_CXX_STANDARD 14)

add_executable(nnlib main.cpp)

add_subdirectory(ann)
add_subdirectory(exceptions)
add_subdirectory(gpu)
add_subdirectory(math)
add_subdirectory(utils)

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
            CUDA_SEPARABLE_COMPILATION ON)
else()
    set_source_files_properties(${SOURCE_FILES} PROPERTIES LANGUAGE CXX)
endif()
