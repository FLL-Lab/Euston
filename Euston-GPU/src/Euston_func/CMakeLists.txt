find_package(CUDAToolkit REQUIRED) # Find CUDA
# EustonCUDA
file(GLOB_RECURSE Euston_CUDA_SOURCE_FILES CONFIGURE_DEPENDS
  "main.cu"
  "gelu.cu"
  "softmax.cu"
  "layernorm.cu"
  "matrix_mul.cu"
  "ckks_evaluator.cu"
)

add_executable(euston_main ${Euston_CUDA_SOURCE_FILES})
target_link_libraries(euston_main PRIVATE lib_utils Phantom 
CUDA::cusolver    # cuSOLVER库
)
target_include_directories(euston_main PRIVATE
	/usr/local/include
#   /usr/local/include/NTL
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/phantom-fhe/include
)