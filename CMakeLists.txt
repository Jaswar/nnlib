cmake_minimum_required(VERSION 3.16)
include(CheckLanguage)

project(nnlib
        DESCRIPTION "GPU-Accelerated Neural Network Library"
        VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 14)
message(STATUS "Using ${CMAKE_CXX_COMPILER_ID} compiler")

add_library(${PROJECT_NAME} STATIC)

add_subdirectory(include)
add_subdirectory(src)

find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})

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
else()
    # If no CUDA, assume only C++
    get_target_property(SOURCE_FILES nnlib SOURCES)
    set_source_files_properties(${SOURCE_FILES} PROPERTIES LANGUAGE CXX)
    if(NOT MSVC)
        set_source_files_properties(${SOURCE_FILES} PROPERTIES COMPILE_OPTIONS -xc++)
    endif()
endif()

include(CheckAVXSupport.cmake)

if(MSVC)
    if (AVX2_SUPPORTED)
        # This flag is added by the compiler anyway, but CLion highlighting does not work without adding
        # this definition manually
        add_definitions(-D__AVX2__)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
        message(STATUS "AVX2 support detected")
    elseif (AVX_SUPPORTED)
        add_definitions(-D__AVX__)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX")
        message(STATUS "AVX support detected")
    else()
        message(STATUS "No SIMD intrinsics detected")
    endif()
else()
    if (AVX2_SUPPORTED)
        # This flag is added by the compiler anyway, but CLion highlighting does not work without adding
        # this definition manually
        add_definitions(-D__AVX2__)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
        message(STATUS "AVX2 support detected")
    elseif (AVX_SUPPORTED)
        add_definitions(-D__AVX__)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
        message(STATUS "AVX support detected")
    else()
        message(STATUS "No SIMD intrinsics detected")
    endif()
endif()

target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)


if(DEFINED CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(
            STATUS
            "CMAKE_INSTALL_PREFIX is not set\n"
            "Default value: ${CMAKE_INSTALL_PREFIX}\n"
            "Will set it to ${CMAKE_SOURCE_DIR}/install"
    )
    set(CMAKE_INSTALL_PREFIX
            "${CMAKE_SOURCE_DIR}/install"
            CACHE PATH "Where the library will be installed to" FORCE
            )
else()
    message(
            STATUS
            "CMAKE_INSTALL_PREFIX was already set\n"
            "Current value: ${CMAKE_INSTALL_PREFIX}"
    )
endif()

set(PUBLIC_HEADERS include/network.h include/layer.h include/matrix.h include/vector.h include/allocation.h
        include/onehot_encode.h include/read.h include/verify.cuh include/convert.h include/activation.h)

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${PUBLIC_HEADERS}")

set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "d")

include(GNUInstallDirs)

install(TARGETS ${PROJECT_NAME}
        EXPORT "${PROJECT_NAME}Targets"
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT "${PROJECT_NAME}Targets"
        FILE "${PROJECT_NAME}Targets.cmake"
        DESTINATION cmake)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        VERSION "${version}"
        COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        INSTALL_DESTINATION cmake)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION cmake
        )

# Try to copy the pdb file in one of the two ways if using MSVC
if (MSVC)
    install(FILES "${CMAKE_SOURCE_DIR}/build/${PROJECT_NAME}.dir/${CMAKE_BUILD_TYPE}/nnlib.pdb"
            DESTINATION lib
            OPTIONAL)

    install(DIRECTORY
            $<TARGET_FILE_DIR:nnlib>/
            DESTINATION lib

            FILES_MATCHING
            PATTERN "*.pdb")
endif()

# generate the export targets for the build tree
export(EXPORT "${PROJECT_NAME}Targets"
        FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}Targets.cmake")

# See if the library is built in test mode and if it is add testing
if (CMAKE_TEST_NNLIB)
    message(STATUS "Building ${PROJECT_NAME} with testing")
    add_subdirectory(test)
endif()

