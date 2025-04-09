
add_library(lib_utils SHARED utils.cpp myread.cpp)

target_link_libraries(lib_utils Eigen3::Eigen SEAL::seal OpenMP::OpenMP_CXX)
target_include_directories(lib_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} )

# Bootstrapping
add_subdirectory(bootstrapping/common)
add_subdirectory(bootstrapping)
add_executable(
    bootstrapping
    bootstrapping.cpp
    NEXUS_func/ckks_evaluator.cpp
	${COMMON_SOURCE_FILES}
    ${BOOTSTRAPPING_SOURCE_FILES}
)

target_include_directories(bootstrapping PUBLIC
    /usr/local/include
    /usr/local/include/NTL
	${CMAKE_CURRENT_SOURCE_DIR}/NEXUS_func
    ${COMMON_HEADER_DIR}
    ${BOOTSTRAPPING_HEADER_DIR}
)

target_link_libraries(bootstrapping PRIVATE ntl gmp m pthread SEAL::seal lib_utils)

add_subdirectory(Euston_func)
add_subdirectory(NEXUS_func)