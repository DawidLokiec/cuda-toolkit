cmake_minimum_required(VERSION 3.25.1)
project(cuda-toolkit LANGUAGES CUDA VERSION 1.0.0)

set(CMAKE_CUDA_STANDARD 23)

add_library(${PROJECT_NAME} src/error.cu src/memory.cu src/gpu_facade.cu)

target_include_directories(${PROJECT_NAME} PUBLIC include/)
# A CMake pattern to have headers that are not seen by the consumers of this library.
# So headers which not have to be visible for consumers should be placed in src/
target_include_directories(${PROJECT_NAME} PRIVATE src)

# We need to explicitly state that we need all CUDA files in the
# this library to be built with as the member functions
# could be called by other libraries and executables
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
