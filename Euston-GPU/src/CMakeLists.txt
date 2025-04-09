add_library(lib_utils SHARED 
	utils.cu 
	myread.cu
)

target_link_libraries(lib_utils Phantom OpenMP::OpenMP_CXX)
target_include_directories(lib_utils PUBLIC 
	${CMAKE_CURRENT_SOURCE_DIR} 
	${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/include
)

# CUDA Boostrapping
file(GLOB_RECURSE NEXUS_CUDA_BOOTSTRAPING_SOURCE_FILES CONFIGURE_DEPENDS
  "bootstrapping.cu"
  "bootstrapping/*.cu"
  "NEXUS_func/ckks_evaluator.cu"
  "../thirdparty/phantom-fhe/src/*.cu"
)

add_executable(bootstrapping ${NEXUS_CUDA_BOOTSTRAPING_SOURCE_FILES})
target_link_libraries(bootstrapping PRIVATE ntl gmp m pthread lib_utils)
target_include_directories(bootstrapping PRIVATE
  /usr/local/include
  /usr/local/include/NTL
  NEXUS_func
  bootstrapping/commmon
  bootstrapping
)


add_subdirectory(Euston_func)
add_subdirectory(NEXUS_func)



