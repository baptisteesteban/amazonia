cmake_minimum_required(VERSION 3.23)
project(amazonia CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75 80 90)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

option(AMAZONIA_BUILD_TESTS "Build the tests" OFF)
option(AMAZONIA_BUILD_DOC "Build the documentation" ON)

include(FetchContent)

add_subdirectory(amazonia)
add_subdirectory(bin)
if (AMAZONIA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif(AMAZONIA_BUILD_TESTS)
if (AMAZONIA_BUILD_DOC)
    add_subdirectory(doc)
endif(AMAZONIA_BUILD_DOC)