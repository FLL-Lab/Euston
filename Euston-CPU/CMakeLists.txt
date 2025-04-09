cmake_minimum_required (VERSION 3.10)

project (Euston) #Eifficient and User-Friendly Secure Transformer Inference

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# Debugging config
option(SHOW_WARNINGS "Enable warnings" OFF)
if (SHOW_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-comment ")
endif()


set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/SEALlibs)
find_package(SEAL REQUIRED)
find_package(Eigen3 REQUIRED)
if(TARGET Eigen3::Eigen)
    get_target_property(_eigen_include_dir Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "Eigen3 Include Path: ${_eigen_include_dir}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Eigen3_CXX_FLAGS}")    
else()
    message(FATAL_ERROR "Eigen3 not found!")

endif()
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "found openmp") 
    # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OPENMP_C_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "openmp not found!")
endif()



add_subdirectory(src)