cmake_minimum_required(VERSION 3.20)

# Must use CUDA 11
if (UNIX)
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.2/bin/nvcc")
endif()

project(EustonTest-CUDA LANGUAGES CXX CUDA)

# CMAKE config
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

# Debugging config
option(SHOW_WARNINGS "Enable warnings" OFF)
if (SHOW_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-comment")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra,-Wpedantic,-Wno-comment")
endif()

# Language config
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA config
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES native) # auto-detect GPU
endif()
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 --default-stream per-thread")

# Print CUDA info
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Compiler ID: ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "CUDA Compiler Version: ${CMAKE_CUDA_COMPILER_VERSION}")


find_package(OpenMP REQUIRED)

# PhantomFHE config
option(PHANTOM_USE_CUDA_PTX "Use CUDA PTX Assembly" ON)
message(STATUS "Use CUDA PTX Assembly: ${PHANTOM_USE_CUDA_PTX}")
if (PHANTOM_USE_CUDA_PTX)
    add_compile_definitions(PHANTOM_USE_CUDA_PTX)
endif()

# PhantomFHE
include_directories(thirdparty/phantom-fhe/include)

file(GLOB_RECURSE PHANTOM_SOURCES CONFIGURE_DEPENDS
	"thirdparty/phantom-fhe/src/*.cu"
)
add_library(Phantom SHARED 
  ${PHANTOM_SOURCES}
)
target_include_directories(Phantom PUBLIC
  /usr/local/include
#   /usr/local/include/NTL
)
# target_link_libraries(Phantom PUBLIC ntl gmp m pthread)
message(STATUS "Phantom built")


add_subdirectory(src)

