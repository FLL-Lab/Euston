#pragma once

#include <fstream>
#include <iostream>
#include <thread>
#include <math.h>
#include <cmath>
#include <vector>
#include <sstream>
#include <algorithm>
#include <limits>
#include <iomanip>
#include <numeric>
#include <omp.h>
#include <chrono>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include "phantom.h"
#include <getopt.h>
using namespace std;
using namespace phantom;
using namespace chrono;

using tensor = vector<double>;
using matrix = vector<tensor>;

#define print_precision 2
static struct option long_options[] = {
	{"batchsize", required_argument, 0, 'b'},
	{"model", required_argument, 0, 'm'},
	{"protocol", required_argument, 0, 'p'},
	{0, 0, 0, 0}
};
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                        \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                          \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)
#define CUSOLVER_CHECK(err)                                                                        \
	do {                                                                                           \
		cusolverStatus_t err_ = (err);                                                             \
		if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
			printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
			throw std::runtime_error("cusolver error");                                            \
		}                                                                                          \
	} while (0)
using tensor = std::vector<double>;
using matrix = std::vector<std::vector<double>>;

// GPU内存管理类
class GPUMemory {
    double* data_ = nullptr;
    size_t size_ = 0;
public:
    explicit GPUMemory(size_t size) : size_(size) {
        CUDA_CHECK(cudaMalloc(&data_, size * sizeof(double)));
    }
    ~GPUMemory() { if(data_) cudaFree(data_); }
    operator double*() { return data_; }
};


matrix mat_add_tensor(const matrix &A, const tensor &B);
matrix mat_add_mat(const matrix& A, const matrix& B);
matrix mat_sub_mat(const matrix& A, const matrix& B);
matrix mat_mul_mat(const matrix &A, const matrix &B);
matrix mat_transpose(const matrix& A);
tensor hadamard_product(const tensor& vec1, const tensor& vec2);
tensor tensor_add_tensor(const tensor& vec1, const tensor& vec2);

matrix mat_add_tensor_gpu(const matrix &A, const tensor &B);
matrix mat_add_mat_gpu(const matrix& A, const matrix& B);
matrix mat_sub_mat_gpu(const matrix& A, const matrix& B);
matrix mat_mul_mat_gpu(const matrix &A, const matrix &B);
matrix mat_transpose_gpu(const matrix& A);
tensor hadamard_product_gpu(const tensor& vec1, const tensor& vec2);
tensor tensor_add_tensor_gpu(const tensor& vec1, const tensor& vec2);
void mat_scale(matrix& mat, double scale);

// Eigen::MatrixXd vectorToEigen(const matrix& mat);
// matrix eigenToVector(const Eigen::MatrixXd& eigenMat);
// void svd_decomposition(const matrix& input, matrix& U, tensor& D, matrix& V);
void svd_decomposition_gpu(const matrix& input, matrix& U, tensor& D, matrix& V);
bool verify_svd(const matrix& original, const matrix& U, tensor& S, const matrix& V);

void save_matrix(const matrix& cpp_vector, string filename);
matrix load_matrix(const string& filename);
void print_error(string message);
void print_tensor(tensor& t, string des, int num);

class Rectime {
public:
	Rectime(){}
	~Rectime(){}
    void start();
    double end();
    double elapsed(string operation);

private:
    time_point<high_resolution_clock> start_time;
    time_point<high_resolution_clock> end_time;
    bool running = false;
};

void plain_rotate(tensor& vec, int positions);