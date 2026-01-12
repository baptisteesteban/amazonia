cmake_minimum_required(VERSION 3.23)
project(amazonia CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75 80 90)

add_subdirectory(bin)