cmake_minimum_required(VERSION 3.24)
project(polysome LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA is optional: the backend is built by default only when the toolkit is found.
find_package(CUDAToolkit QUIET)
option(POLYSOME_CUDA "Build the CUDA backend" ${CUDAToolkit_FOUND})
if(POLYSOME_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)  # Turing, RTX 2060 Super
  endif()
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  enable_language(CUDA)
endif()
message(STATUS "polysome: build type ${CMAKE_BUILD_TYPE}, CUDA ${POLYSOME_CUDA}")

# Unit tests: every tests/*.cpp goes into one binary, tests/test_main.cpp holds the runner.
file(GLOB POLYSOME_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/*.cpp)
add_executable(polysome_tests ${POLYSOME_TEST_SOURCES})
target_include_directories(polysome_tests PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_compile_options(polysome_tests PRIVATE -Wall -Wextra -Wpedantic)
# Makefiles do not relink when a source is only removed, which would leave stale tests in the
# binary. The list file is rewritten only when the glob changes, and relinks through LINK_DEPENDS.
file(CONFIGURE OUTPUT ${CMAKE_BINARY_DIR}/test_sources.txt CONTENT "${POLYSOME_TEST_SOURCES}")
set_target_properties(polysome_tests PROPERTIES LINK_DEPENDS ${CMAKE_BINARY_DIR}/test_sources.txt)

enable_testing()
add_test(NAME unit COMMAND polysome_tests)
