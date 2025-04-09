find_package(CUDAToolkit REQUIRED) # Find CUDA
# NexusCUDA
file(GLOB_RECURSE NEXUS_CUDA_SOURCE_FILES CONFIGURE_DEPENDS
  "main.cu"
  "gelu.cu"
  "softmax.cu"
  "layer_norm.cu"
  "matrix_mul.cu"
  "matrix_mul_p.cu"
  "ckks_evaluator.cu"
)
add_executable(nexus_main ${NEXUS_CUDA_SOURCE_FILES})
target_link_libraries(nexus_main PRIVATE lib_utils Phantom CUDA::cusolver)
target_include_directories(nexus_main PRIVATE
  /usr/local/include
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/include
)