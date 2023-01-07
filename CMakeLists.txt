cmake_minimum_required(VERSION 3.24)
project(cuda_toolkit LANGUAGES CXX CUDA VERSION 0.1.0)

set(CMAKE_CUDA_STANDARD 23)

add_library(cuda_toolkit src/library.cu)

target_include_directories(cuda_toolkit PUBLIC include/)
# A CMake pattern to have headers that are not seen by the consumers of this library.
# So headers which not have to be visible for consumers should be placed in src/
target_include_directories(cuda_toolkit PRIVATE src ${CMAKE_CURRENT_BINARY_DIR})

# We need to explicitly state that we need all CUDA files in the
# this library to be built with as the member functions
# could be called by other libraries and executables
set_target_properties(cuda_toolkit PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
