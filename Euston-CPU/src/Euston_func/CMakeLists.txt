add_library(lib_euston_func
	ckks_evaluator.cpp
    matrix_mul.cpp
	layernorm.cpp
    softmax.cpp
	gelu.cpp
)
target_link_libraries(lib_euston_func lib_utils)
target_include_directories(lib_euston_func PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(euston_main "func_main.cpp")
target_link_libraries(euston_main lib_euston_func)