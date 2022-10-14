cmake_minimum_required (VERSION 3.18)

project (GasSimulation CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(lib/glfw)
add_subdirectory(lib/glad)
add_subdirectory(lib/glm)

add_executable (GasSimulation "cuda.cu" "cuda.cuh" "main.cpp" "Timer.cpp" "Timer.h")

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(GasSimulation glfw glad glm)

set_target_properties(GasSimulation PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)