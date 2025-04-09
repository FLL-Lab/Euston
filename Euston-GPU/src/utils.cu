#include "utils.cuh"

tensor hadamard_product(const tensor& vec1, const tensor& vec2)
{
    if (vec1.size() != vec2.size()) {
        throw invalid_argument("Vectors must be of the same size for Hadamard product.");
    }
    tensor result(vec1.size());
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] * vec2[i];
    }
    return result;
}

// GPU内核实现逐元素运算
__global__ void hadamard_product_kernel(
    const double* vec1, const double* vec2, 
    double* result, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = vec1[idx] * vec2[idx];
    }
}

// 调用示例（需提前分配显存）：
tensor hadamard_product_gpu(const tensor& vec1, const tensor& vec2) {
	tensor result;
    int size = vec1.size();
    const int block_size = 256;
    hadamard_product_kernel<<<(size+block_size-1)/block_size, block_size>>>(
        vec1.data(), vec2.data(), result.data(), size);
	return result;
}

tensor tensor_add_tensor(const tensor& vec1, const tensor& vec2)
{
    if (vec1.size() != vec2.size()) {
        throw invalid_argument("Vectors must be of the same size for element-wise addition.");
    }
    tensor result(vec1.size());
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] + vec2[i];
    }
    return result;
}
__global__ void add_kernel(const double* a, const double* b, double* c, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}
// 张量加法
tensor tensor_add_tensor_gpu(const tensor& vec1, const tensor& vec2) {
    if (vec1.size() != vec2.size())
        throw std::invalid_argument("Vector size mismatch");

    const size_t size = vec1.size();
    tensor result(size);
    
    GPUMemory d_vec1(size), d_vec2(size), d_result(size);
    
    CUDA_CHECK(cudaMemcpy(d_vec1, vec1.data(), size*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec2, vec2.data(), size*sizeof(double), cudaMemcpyHostToDevice));

    const int block_size = 256;
    const dim3 grid_size((size + block_size - 1) / block_size);
    
    add_kernel<<<grid_size, block_size>>>(d_vec1, d_vec2, d_result, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(result.data(), d_result, size*sizeof(double), cudaMemcpyDeviceToHost));
    return result;
}
matrix mat_add_tensor(const matrix &A, const tensor &B)
{
    size_t numRows = A.size();
    size_t numCols = A[0].size();
    if (B.size() != numCols) {
        throw out_of_range("Tensor B size does not match the number of columns in matrix A.");
    }
    matrix result(numRows, tensor(numCols, 0.0));
    for (size_t i = 0; i < numRows; ++i) {
		
        for (size_t j = 0; j < numCols; ++j) {
            result[i][j] = A[i][j] + B[j];
        }
    }
    return result;
}
__global__ void mat_add_tensor_kernel(const double* A, const double* B, double* C, size_t rows, size_t cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols)
        C[row*cols + col] = A[row*cols + col] + B[col];
};
// 矩阵与张量广播加法
matrix mat_add_tensor_gpu(const matrix &A, const tensor &B) {
    const size_t rows = A.size(), cols = A[0].size();
    if (B.size() != cols) throw std::out_of_range("Dimension mismatch");

    matrix result(rows, tensor(cols));
    const size_t mat_size = rows * cols;

    GPUMemory d_A(mat_size), d_B(cols), d_C(mat_size);
    
    // 展平矩阵为列主序
    std::vector<double> flat_A(mat_size);
    for (size_t i = 0; i < rows; ++i)
        std::copy(A[i].begin(), A[i].end(), flat_A.begin() + i * cols);
    
    CUDA_CHECK(cudaMemcpy(d_A, flat_A.data(), mat_size*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), cols*sizeof(double), cudaMemcpyHostToDevice));

    const dim3 block(16, 16);
    const dim3 grid((cols + 15)/16, (rows + 15)/16);
    
    mat_add_tensor_kernel<<<grid, block>>>(d_A, d_B, d_C, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 转换回行主序
    std::vector<double> flat_C(mat_size);
    CUDA_CHECK(cudaMemcpy(flat_C.data(), d_C, mat_size*sizeof(double), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < rows; ++i)
        std::copy(flat_C.begin() + i*cols, flat_C.begin() + (i+1)*cols, result[i].begin());
    
    return result;
}
matrix mat_add_mat(const matrix& A, const matrix& B) 
{
	auto start = chrono::high_resolution_clock::now();
    if (A.size() != B.size() || A.empty() || B.empty() || A[0].size() != B[0].size()) {
        throw invalid_argument("Matrices dimensions must be the same for addition.");
    }

    matrix result(A.size(), tensor(A[0].size(), 0.0));

    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[i].size(); ++j) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
	auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    // cout << "mat_add_mat Elapsed time: " << elapsed.count() << " s" << endl;
    return result;
}
matrix mat_sub_mat(const matrix& A, const matrix& B) 
{

    if (A.size() != B.size() || A.empty() || B.empty() || A[0].size() != B[0].size()) {
        throw invalid_argument("Matrices dimensions must be the same for addition.");
    }

    matrix result(A.size(), tensor(A[0].size(), 0.0));
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[i].size(); ++j) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }
    return result;
}
__global__ void matrix_elementwise_kernel(const double* a, const double* b, double* c, size_t rows, size_t cols, bool IsAdd) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        const int idx = row * cols + col;
        c[idx] = IsAdd ? (a[idx] + b[idx]) : (a[idx] - b[idx]);
    }
}
// 矩阵加减法模板
template<bool IsAdd>
matrix matrix_elementwise_gpu(const matrix& A, const matrix& B) {
    const size_t rows = A.size(), cols = A[0].size();
    if (rows != B.size() || cols != B[0].size())
        throw std::invalid_argument("Matrix dimension mismatch");

    matrix result(rows, tensor(cols));
    const size_t mat_size = rows * cols;

    GPUMemory d_A(mat_size), d_B(mat_size), d_C(mat_size);
    
    // 展平矩阵为列主序
    std::vector<double> flat_A(mat_size), flat_B(mat_size);
    for (size_t i = 0; i < rows; ++i) {
        std::copy(A[i].begin(), A[i].end(), flat_A.begin() + i * cols);
        std::copy(B[i].begin(), B[i].end(), flat_B.begin() + i * cols);
    }
    
    CUDA_CHECK(cudaMemcpy(d_A, flat_A.data(), mat_size*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, flat_B.data(), mat_size*sizeof(double), cudaMemcpyHostToDevice));

    const dim3 block(16, 16);
    const dim3 grid((cols + 15)/16, (rows + 15)/16);
    
    auto kernel = [](const double* a, const double* b, double* c, size_t rows, size_t cols) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < rows && col < cols) {
            const int idx = row * cols + col;
            c[idx] = IsAdd ? (a[idx] + b[idx]) : (a[idx] - b[idx]);
        }
    };
    
    matrix_elementwise_kernel<<<grid, block>>>(d_A, d_B, d_C, rows, cols, IsAdd);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 转换回行主序
    std::vector<double> flat_C(mat_size);
    CUDA_CHECK(cudaMemcpy(flat_C.data(), d_C, mat_size*sizeof(double), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < rows; ++i)
        std::copy(flat_C.begin() + i*cols, flat_C.begin() + (i+1)*cols, result[i].begin());
    
    return result;
}

matrix mat_add_mat_gpu(const matrix& A, const matrix& B) {
    return matrix_elementwise_gpu<true>(A, B);
}

matrix mat_sub_mat_gpu(const matrix& A, const matrix& B) {
    return matrix_elementwise_gpu<false>(A, B);
}
matrix mat_mul_mat(const matrix &A, const matrix &B)
{
    size_t numRowsA = A.size();
    size_t numColsA = A[0].size();
	size_t numRowsB = B.size();
    size_t numColsB = B[0].size();
    
    matrix result(numRowsA, tensor(numColsB, 0.0));
	if(numColsA != numRowsB){
		cerr << "Dimensions not match when mat_mul_mat." << endl;
		return result;
	}

   	for (size_t i = 0; i < numRowsA; ++i) {
        for (size_t k = 0; k < numColsA; ++k) {
            double valA = A[i][k];
            for (size_t j = 0; j < numColsB; ++j) {
                result[i][j] += valA * B[k][j];
            }
        }
    }
    return result;
}
matrix mat_mul_mat_gpu(const matrix &A, const matrix &B) {
    const size_t M = A.size(), K = A[0].size(), N = B[0].size();
    if (K != B.size()) throw std::invalid_argument("Matrix dimensions mismatch");

    matrix result(M, tensor(N));
    const size_t a_size = M * K;
    const size_t b_size = K * N;
    const size_t c_size = M * N;

    // 展平矩阵为行主序
    std::vector<double> flat_A(a_size), flat_B(b_size), flat_C(c_size);
    for (size_t i = 0; i < M; ++i)
        std::copy(A[i].begin(), A[i].end(), flat_A.begin() + i*K);
    for (size_t i = 0; i < K; ++i)
        std::copy(B[i].begin(), B[i].end(), flat_B.begin() + i*N);

    GPUMemory d_A(a_size), d_B(b_size), d_C(c_size);
    
    CUDA_CHECK(cudaMemcpy(d_A, flat_A.data(), a_size*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, flat_B.data(), b_size*sizeof(double), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const double alpha = 1.0, beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha,
                d_B, N,
                d_A, K,
                &beta, d_C, N);
    
    CUDA_CHECK(cudaMemcpy(flat_C.data(), d_C, c_size*sizeof(double), cudaMemcpyDeviceToHost));
    
    // 转换回二维结果
    for (size_t i = 0; i < M; ++i)
        std::copy(flat_C.begin() + i*N, flat_C.begin() + (i+1)*N, result[i].begin());
    
    cublasDestroy(handle);
    return result;
}
void mat_scale(matrix& mat, double scale)
{
	for (size_t i = 0; i < mat.size(); ++i) {
        for (size_t j = 0; j < mat[0].size(); ++j) {
            mat[i][j] *= scale;
        }
    }
}
matrix mat_transpose(const matrix& A)
{
    size_t rows = A.size();
    size_t cols = A.empty() ? 0 : A[0].size();
	matrix result;
    try {
		result.resize(cols);
		
		for(int i = 0; i < cols; i++)
		{
			result[i].resize(rows);
		}
	} catch (const bad_alloc& e) {
		cerr << "Memory allocation failed: " << e.what() << endl;
		throw;
	}

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j][i] = A[i][j];
        }
    }
    return result;
}
__global__ void transpose_kernel(const double* input, double* output, size_t rows, size_t cols) {
    __shared__ double block[32][32+1];  // 避免共享内存bank冲突
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < cols && y < rows) {
        block[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    __syncthreads();
    
    int tx = blockIdx.y * blockDim.y + threadIdx.x;
    int ty = blockIdx.x * blockDim.x + threadIdx.y;
    if (tx < rows && ty < cols) {
        output[ty * rows + tx] = block[threadIdx.x][threadIdx.y];
    }
}

matrix mat_transpose_gpu(const matrix& A) {
    const size_t rows = A.size(), cols = A[0].size();
    const size_t mat_size = rows * cols;
    
    // 展平矩阵为行主序
    std::vector<double> flat_A(mat_size), flat_AT(mat_size);
    for (size_t i = 0; i < rows; ++i)
        std::copy(A[i].begin(), A[i].end(), flat_A.begin() + i*cols);

    GPUMemory d_A(mat_size), d_AT(mat_size);
    CUDA_CHECK(cudaMemcpy(d_A, flat_A.data(), mat_size*sizeof(double), cudaMemcpyHostToDevice));

    dim3 block(32, 32);
    dim3 grid((cols + 31)/32, (rows + 31)/32);
    transpose_kernel<<<grid, block>>>(d_A, d_AT, rows, cols);
    
    CUDA_CHECK(cudaMemcpy(flat_AT.data(), d_AT, mat_size*sizeof(double), cudaMemcpyDeviceToHost));

    // 转换回二维转置矩阵
    matrix AT(cols, tensor(rows));
    for (size_t i = 0; i < cols; ++i)
        std::copy(flat_AT.begin() + i*rows, flat_AT.begin() + (i+1)*rows, AT[i].begin());
    
    return AT;
}
matrix readMatrix(const string &filename, int rows, int cols) 
{
  matrix mat(rows, tensor(cols));
  ifstream file(filename);

  if (!file.is_open()) {
    cerr << "Can not open file: " << filename << endl;
    return mat;
  }

  string line;
  for (int i = 0; i < rows; ++i) {
    if (getline(file, line)) {
      istringstream iss(line);
      for (int j = 0; j < cols; ++j) {
        if (!(iss >> mat[i][j])) {
          cerr << "read error: " << filename << " (row: " << i << ", column: " << j << ")" << endl;
        }
      }
    }
  }

  file.close();
  return mat;
}

// // 将 matrix 转换为 Eigen::MatrixXd
// Eigen::MatrixXd vectorToEigen(const matrix& mat) {
//     int rows = mat.size();
//     int cols = mat[0].size();
//     Eigen::MatrixXd eigenMat(rows, cols);
//     for (int i = 0; i < rows; ++i) {
//         for (int j = 0; j < cols; ++j) {
//             eigenMat(i, j) = mat[i][j];
//         }
//     }
//     return eigenMat;
// }

// matrix eigenToVector(const Eigen::MatrixXd& eigenMat) {
//     int rows = eigenMat.rows();
//     int cols = eigenMat.cols();
//     matrix mat(rows, tensor(cols));
//     for (int i = 0; i < rows; ++i) {
//         for (int j = 0; j < cols; ++j) {
//             mat[i][j] = eigenMat(i, j);
//         }
//     }
//     return mat;
// }

// void svd_decomposition(const matrix& input, matrix& U, tensor& D, matrix& V) {
//     Eigen::MatrixXd eigenMat = vectorToEigen(input);

//     Eigen::JacobiSVD<Eigen::MatrixXd> svd(eigenMat, Eigen::ComputeThinU | Eigen::ComputeThinV);
//     Eigen::MatrixXd eigenU = svd.matrixU();
//     Eigen::MatrixXd eigenV = svd.matrixV();
//     Eigen::VectorXd eigenD = svd.singularValues();

//     U = eigenToVector(eigenU);
//     V = eigenToVector(eigenV);
// 	V = mat_transpose(V);
//     D = tensor(eigenD.data(), eigenD.data() + eigenD.size());
// }
void matrix_to_vector(const matrix& A, std::vector<double>& vec)
{
    for (int j = 0; j < A[0].size(); j++)
    {
        for (int i = 0; i < A.size(); i++)
        {
            vec.push_back(A[i][j]);
        }
    }
}
void vector_to_matrix(const std::vector<double>& vec, matrix& A, int rows, int cols)
{
	A.resize(rows, std::vector<double>(cols));
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			A[i][j] = vec[j * rows + i];
		}
	}
}
void svd_decomposition_gpu(const matrix& myinput, matrix& myU, tensor& myD, matrix& myV)
{
	matrix input;
	int transposed = 0;
	if(myinput.size() < myinput[0].size()) {
		input = mat_transpose(myinput);
		transposed = 1;
	} else {
		input = myinput;
	}
	std::vector<double> A;
	matrix_to_vector(input, A);
	
	cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

	const int m = input.size();
	const int n = input[0].size();
	const int lda = m;  // lda >= m
	const int ldu = m;  // ldu >= m
	const int ldvt = n; // ldvt >= n if jobu = 'A'

	std::vector<double> U(ldu * m, 0);  /* m-by-m unitary matrix, left singular vectors  */
    std::vector<double> VT(ldvt * n, 0); /* n-by-n unitary matrix, right singular vectors */
    std::vector<double> S(n, 0);        /* numerical singular value */

	int info_gpu = 0;                                  /* host copy of error info */

    double *d_A = nullptr;
    double *d_S = nullptr;  /* singular values */
    double *d_U = nullptr;  /* left singular vectors */
    double *d_VT = nullptr; /* right singular vectors */
    double *d_W = nullptr;  /* W = S*VT */

    int *devInfo = nullptr;

    int lwork = 0; /* size of workspace */
    double *d_work = nullptr;
    double *d_rwork = nullptr;

	/* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

	/* step 2: copy A to device */
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_S), sizeof(double) * S.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_U), sizeof(double) * U.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_VT), sizeof(double) * VT.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(double) * lda * n));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&devInfo), sizeof(int)));

	CUDA_CHECK(
        cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice, stream));

    /* step 3: query working space of SVD */
    CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(cusolverH, m, n, &lwork));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));
	/* step 4: compute SVD */
	signed char jobu = 'A';  // all m columns of U
    signed char jobvt = 'A'; // all n rows of VT
    CUSOLVER_CHECK(cusolverDnDgesvd(cusolverH, jobu, jobvt, m, n, d_A, lda,
                                    d_S, d_U, ldu, d_VT, ldvt,
                                    d_work, lwork, d_rwork, devInfo));

	CUDA_CHECK(cudaMemcpyAsync(U.data(), d_U, sizeof(double) * U.size(), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(VT.data(), d_VT, sizeof(double) * VT.size(), cudaMemcpyDeviceToHost,
							   stream));
	CUDA_CHECK(cudaMemcpyAsync(S.data(), d_S, sizeof(double) * S.size(), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));

	if(transposed) {
		vector_to_matrix(U, myV, m, n);
		myV = mat_transpose(myV);
		vector_to_matrix(VT, myU, n, n);
		myU = mat_transpose(myU);
		myD = S;
	} else {
		vector_to_matrix(U, myU, m, n);
		vector_to_matrix(VT, myV, n, n);
		myD = S;
	}
	/* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_VT));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(devInfo));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_rwork));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));
}

void print_my_matrix(matrix A) {
	std::cout << std::fixed << std::setprecision(3);
    for (int i = 0; i < A.size(); i++) {
        for (int j = 0; j < A[0].size(); j++) {
            std::cout << A[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// bool verify_svd(const matrix& original, const matrix& U, const tensor& S, const matrix& V) {
//     Eigen::MatrixXd eigenU = vectorToEigen(U);
//     Eigen::MatrixXd eigenV = vectorToEigen(V);
//     Eigen::VectorXd eigenS = Eigen::Map<const Eigen::VectorXd>(S.data(), S.size());
    
//     Eigen::MatrixXd sigma = Eigen::MatrixXd::Zero(eigenS.size(), eigenS.size());
//     for (int i = 0; i < eigenS.size(); ++i) {
//         sigma(i, i) = eigenS(i);
//     }
    
//     Eigen::MatrixXd reconstructed = eigenU * sigma * eigenV.transpose();
//     matrix reconstructedVec = eigenToVector(reconstructed);

//     const double tolerance = 1e-6; 
//     for (size_t i = 0; i < original.size(); ++i) {
//         for (size_t j = 0; j < original[0].size(); ++j) {
//             if (abs(original[i][j] - reconstructedVec[i][j]) > tolerance) {
//                 return false;
//             }
//         }
//     }
//     return true;
// }

void save_matrix(const matrix& cpp_vector, string filename)
{
	ofstream file(filename);
	 if (!file.is_open()) {
        cerr << "Error opening file for writing: " << filename << endl;
        return;
    }
    for (const auto& row : cpp_vector) {
        for (const auto& val : row) {
            file << fixed << setprecision(6) << val << " ";
        }
        file << "\n";
    }
    file.close();
}
matrix load_matrix(const string& filename) 
{
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file for reading: " << filename << endl;
        return matrix();
    }

    matrix mat;
    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        tensor row;
        double val;
        while (iss >> val) {
            row.push_back(val);
        }
        mat.push_back(row);
    }

    file.close();
    return mat;
}

void print_error(string message)
{
	cerr << "Error: " + message << endl;
}

void print_tensor(tensor& t, string des, int num)
{
	cout << des;
	for(int i = 0; i < num; i++)
	{
		cout << " " << t[i] ;
	}
	cout << endl;
}

void Rectime::start() {
	start_time = chrono::high_resolution_clock::now();
	running = true;
}

double Rectime::end() {
	if (running) {
		end_time = chrono::high_resolution_clock::now();
		running = false;
	}
	return chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
}

double Rectime::elapsed(string operation) {
	chrono::time_point<chrono::high_resolution_clock> end;
	if (running) {
		end = chrono::high_resolution_clock::now();
	} else {
		end = end_time;
	}
	auto elapsed_time = chrono::duration_cast<milliseconds>(end - start_time).count();
	cout << operation + " Elapsed time: " << fixed << setprecision(6) << elapsed_time << " milliseconds\n";
	start();
	return elapsed_time;
}

void plain_rotate(tensor& vec, int positions) {
    if (vec.empty()) {
        return;
    }
    int n = vec.size();
    positions = ((positions % n) + n) % n;

    if (positions > 0) {
        rotate(vec.begin(), vec.begin() + positions, vec.end());
    } else if (positions < 0) {
        rotate(vec.rbegin(), vec.rbegin() - positions, vec.rend());
    }
}