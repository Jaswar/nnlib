cmake_minimum_required(VERSION 3.20)

project(mnist_nnlib LANGUAGES CXX)
#set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_PREFIX_PATH "C:/Users/janwa/CLionProjects/nnlib/install")
find_package(nnlib CONFIG REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

add_compile_options(-pthread)

target_link_libraries(${PROJECT_NAME} PRIVATE nnlib)
