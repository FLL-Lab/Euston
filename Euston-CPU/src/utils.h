#ifndef UTILS_H
#define UTILS_H

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
#include <eigen3/Eigen/Dense>
#include <seal/seal.h>
#include <seal/util/uintarith.h>
#include <getopt.h>
using namespace std;
using namespace seal;
using namespace seal::util;
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
matrix mat_add_tensor(const matrix &A, const tensor &B);
matrix mat_add_mat(const matrix& A, const matrix& B);
matrix mat_sub_mat(const matrix& A, const matrix& B);
matrix mat_mul_mat(const matrix &A, const matrix &B);
matrix mat_transpose(const matrix& A);
tensor hadamard_product(const tensor& vec1, const tensor& vec2);
tensor tensor_add_tensor(const tensor& vec1, const tensor& vec2);
void mat_scale(matrix& mat, double scale);

Eigen::MatrixXd vectorToEigen(const matrix& mat);
matrix eigenToVector(const Eigen::MatrixXd& eigenMat);
void svd_decomposition(const matrix& input, matrix& U, tensor& D, matrix& V);
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
#endif