cmake_minimum_required(VERSION 3.24)
project(cuda_toolkit LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(cuda_toolkit src/main.cpp)

target_include_directories(cuda_toolkit PUBLIC include/)
# A CMake pattern to have headers that are not seen by the consumers of this library.
# So headers which not have to be visible for consumers should be placed in src/
target_include_directories(cuda_toolkit PRIVATE src ${CMAKE_CURRENT_BINARY_DIR})

if (MSVC)
    # PUBLIC = use this compile options also while compiling the project simulation-engine
    # warning level 4 and all warnings as errors
    target_compile_options(cuda_toolkit PUBLIC /W4 /WX)
else()
    # PUBLIC = use this compile options also while compiling the project simulation-engine
    # lots of warnings and all warnings as errors
    target_compile_options(cuda_toolkit PUBLIC -Wall -Wextra -pedantic -Werror)
endif()
