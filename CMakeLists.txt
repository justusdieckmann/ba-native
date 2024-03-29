﻿cmake_minimum_required (VERSION 3.18)

project (GasSimulation CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

if(NOT DEFINED DOGRAPHICS)
    set(DOGRAPHICS 1)
endif()

if(DOGRAPHICS)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(lib/glfw)
add_subdirectory(lib/glad)
endif()

add_subdirectory(lib/glm)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} include)

if(DOGRAPHICS)
add_executable(GasSimulation "cuda.cu" "cuda.cuh" "main.cpp" "Timer.cpp" "Timer.h" include/array.h include/vec3.h)

target_link_libraries(GasSimulation glfw glad glm)

set_target_properties(GasSimulation PROPERTIES
        CUDA_ARCHITECTURES "52;62;86")
endif()

add_executable(test "cuda.cu" "cuda.cuh" "test.cpp" "Timer.cpp" "Timer.h")

target_link_libraries(test glm)

set_target_properties(test PROPERTIES
        CUDA_ARCHITECTURES "52;62;86")