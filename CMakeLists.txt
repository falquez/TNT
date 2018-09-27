cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(TNT
  DESCRIPTION "Tensor Network Toolkit"
  VERSION 0.1
  LANGUAGES CXX
  )

# Main project language
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Save libs and executables in the same place
set(LIBRARY_OUTPUT_PATH    "${CMAKE_BINARY_DIR}/lib" CACHE PATH "Output directory for libraries" )
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Output directory for applications" )

# Find Boost
find_package(Boost COMPONENTS system filesystem REQUIRED)
add_library(boost INTERFACE IMPORTED)

# Local libraries
include_directories($ENV{HOME}/dist/include)
link_directories($ENV{HOME}/dist/lib)

enable_testing()

# TNT library
add_subdirectory(TNT)
# Examples
add_subdirectory(examples)

