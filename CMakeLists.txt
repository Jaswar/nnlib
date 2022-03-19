cmake_minimum_required(VERSION 3.20)
include(CheckLanguage)

project(nnlib)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES math/matrix.cpp math/matrix.h main.cpp exceptions/size_mismatch_exception.cpp exceptions/size_mismatch_exception.h utils/allocation.cpp utils/allocation.h math/vector.cpp math/vector.h ann/network.cpp ann/network.h ann/layer.cpp ann/layer.h ann/activation.cu ann/activation.cuh utils/read.cpp utils/read.h gpu/verify.cuh gpu/verify.cu math/convert.cpp math/convert.h utils/onehot_encode.cpp utils/onehot_encode.h ann/backpropagation.cu ann/backpropagation.cuh gpu/allocation_gpu.cu gpu/allocation_gpu.cuh gpu/assert.cuh exceptions/different_data_location_exception.cpp exceptions/different_data_location_exception.h math/vector_operations.cu math/vector_operations.cuh math/matrix_operations.cu math/matrix_operations.cuh)

add_executable(nnlib ${SOURCE_FILES})

check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    set(CMAKE_CUDA_STANDARD 14)

    # Add flag to specify the architecture of the GPU (compute capability)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50")

    set_target_properties(nnlib PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)
else()
    set_source_files_properties(${SOURCE_FILES} PROPERTIES LANGUAGE CXX)
endif()
