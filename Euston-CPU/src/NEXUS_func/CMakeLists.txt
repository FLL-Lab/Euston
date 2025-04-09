add_library(lib_nexus_func
	gelu.cpp
	layer_norm.cpp
	ckks_evaluator.cpp
	softmax.cpp
	matrix_mul.cpp
)
target_include_directories(lib_nexus_func PUBLIC 
	/usr/local/include
	${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(lib_nexus_func lib_utils)


add_executable(nexus_main "main.cpp")
target_link_libraries(nexus_main lib_nexus_func)